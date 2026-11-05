"""Automatic local candidates and deterministic, non-learned 2D refinement.

Only HU, binary liver support and physical geometry enter this module. Tumor
references are intentionally absent. Context items are (offset_mm, HU, liver).
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

METHODS = ('local_growth', 'local_contour', 'local_graphcut')
CONNECTIVITY = np.ones((3, 3), bool)
DEFAULTS = dict(method='local_growth', high=2., polarity='dark', context=False,
                refinement=2., sigma_mm=.6, background_mm=(3., 6., 12.),
                noise_floor_hu=5., low=.5, separation_mm=2., margin_mm=5.,
                window_mm=5., max_iterations=150)


def validate_settings(settings):
    if set(settings) - set(DEFAULTS):
        raise ValueError(f'Unknown local settings: {set(settings)-set(DEFAULTS)}')
    cfg = {**DEFAULTS, **settings}
    if cfg['method'] not in METHODS or cfg['polarity'] not in ('dark', 'both'):
        raise ValueError('Invalid local method/polarity')
    if not isinstance(cfg['context'], (bool, np.bool_)):
        raise ValueError('context must be boolean')
    for key in ('high', 'refinement', 'sigma_mm', 'noise_floor_hu', 'low',
                'separation_mm', 'margin_mm', 'window_mm'):
        if isinstance(cfg[key], bool) or not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if cfg['high'] <= cfg['low']:
        raise ValueError('high must exceed low')
    scales = np.asarray(cfg['background_mm'], float)
    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('background_mm requires three positive physical scales')
    if isinstance(cfg['max_iterations'], bool) or not isinstance(cfg['max_iterations'], int) or cfg['max_iterations'] < 1:
        raise ValueError('max_iterations must be a positive integer')
    cfg['background_mm'] = tuple(map(float, scales))
    return cfg


def configurations():
    records = []
    for high, polarity, context, method in product((1.5, 2., 2.5), ('dark', 'both'), (False, True), METHODS):
        for refinement in ((1.5, 2., 2.5) if method == 'local_growth' else (.1, .3, .6)):
            cfg = validate_settings(dict(high=high, polarity=polarity, context=context,
                                         method=method, refinement=refinement))
            records.append(dict(configuration_id=f'local_{len(records):03d}', settings=cfg))
    return records


def _inputs(hu, liver, spacing):
    hu, liver, spacing = np.asarray(hu, float), np.asarray(liver), np.asarray(spacing, float)
    if hu.ndim != 2 or hu.shape != liver.shape or not np.isfinite(hu).all() or not np.isin(liver, (0, 1)).all():
        raise ValueError('Expected aligned finite 2D HU and binary liver')
    if spacing.shape not in ((2,), (3,)) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Expected positive physical spacing')
    return hu, liver.astype(bool), spacing[:2]


def physical_disk(radius, spacing):
    size = np.ceil(radius / np.asarray(spacing)).astype(int)
    x, y = np.ogrid[-size[0]:size[0]+1, -size[1]:size[1]+1]
    return (x*spacing[0])**2 + (y*spacing[1])**2 <= radius**2 + 1e-12


def masked_gaussian(image, liver, sigma_mm, spacing):
    weight = ndi.gaussian_filter(liver.astype(float), sigma_mm / spacing)
    value = ndi.gaussian_filter(np.where(liver, image, 0.), sigma_mm / spacing)
    return np.divide(value, weight, out=np.zeros_like(value), where=weight > 1e-12)


def robust(values, floor):
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Missing finite statistics')
    median = float(np.median(values))
    return median, max(floor, float(1.4826*np.median(np.abs(values-median))))


def prepare_maps(hu, liver, spacing, settings):
    cfg = validate_settings(settings)
    hu, liver, spacing = _inputs(hu, liver, spacing)
    smooth = masked_gaussian(hu, liver, cfg['sigma_mm'], spacing)
    scale = robust(smooth[liver], cfg['noise_floor_hu'])[1] if liver.any() else cfg['noise_floor_hu']
    contrasts = np.stack([(masked_gaussian(smooth, liver, r, spacing)-smooth)/scale
                          for r in cfg['background_mm']])
    return dict(image=smooth, liver=liver, spacing=spacing, scale=scale,
                scores=dict(dark=np.maximum(contrasts.max(0), 0)*liver,
                            bright=np.maximum((-contrasts).max(0), 0)*liver),
                scales=dict(dark=contrasts.argmax(0), bright=(-contrasts).argmax(0)),
                preparation={k: cfg[k] for k in ('sigma_mm', 'background_mm', 'noise_floor_hu')})


def context_indices(index, depth, spacing_z):
    return [j for j in range(max(0, index-2), min(depth, index+3))
            if j != index and abs(j-index)*spacing_z <= 5. + 1e-9]


def context_from_volume(volume, liver, index, spacing):
    """Shared GUI/batch provider; binary support only, never tumor labels."""
    if volume.shape != liver.shape or volume.ndim != 3 or not np.isin(liver, (0, 1)).all():
        raise ValueError('Expected aligned volume and binary liver support')
    return [((j-index)*spacing[2], volume[:, :, j], liver[:, :, j])
            for j in context_indices(index, volume.shape[2], spacing[2])]


def prepare_context_maps(context, shape, spacing, settings):
    result = []
    seen = set()
    for offset, hu, liver in context:
        if not np.isfinite(offset) or offset == 0 or abs(offset) > 5.+1e-9 or offset in seen:
            raise ValueError('Invalid/duplicate neighbor offset')
        if np.asarray(hu).shape != shape:
            raise ValueError('Neighbor shape mismatch')
        seen.add(offset)
        result.append((float(offset), prepare_maps(hu, liver, spacing, settings)))
    if sum(x < 0 for x in seen) > 2 or sum(x > 0 for x in seen) > 2:
        raise ValueError('At most two neighbors per side')
    return result


@dataclass
class Candidate:
    identifier: int
    polarity: str
    box: tuple
    seeds: np.ndarray
    support: np.ndarray
    origins: list


def build_candidates(prepared, settings, neighbors=None):
    cfg = validate_settings(settings)
    if any(cfg[k] != v for k, v in prepared['preparation'].items()):
        raise ValueError('Prepared maps/settings mismatch')
    if cfg['context'] and neighbors is None:
        raise ValueError('Local context requested but not supplied; pass [] at a volume boundary')
    liver, spacing = prepared['liver'], prepared['spacing']
    scores, candidates = {}, []
    for polarity in ('dark', 'bright') if cfg['polarity'] == 'both' else ('dark',):
        score = prepared['scores'][polarity].copy()
        if cfg['context'] and neighbors:
            weights = np.zeros_like(score); weighted = np.zeros_like(score)
            for distance, neighbor in neighbors:
                weight = np.exp(-abs(distance)/2.5)*neighbor['liver']
                weights += weight; weighted += weight*neighbor['scores'][polarity]
            mean = np.divide(weighted, weights, out=score.copy(), where=weights > 0)
            score = (.75*score+.25*mean)*liver
        scores[polarity] = score
        local_max = ndi.maximum_filter(score, footprint=physical_disk(cfg['separation_mm'], spacing))
        peaks = liver & (score >= cfg['high']) & (score == local_max)
        plateaus, _ = ndi.label(peaks, CONNECTIVITY)
        points = []
        for label_id, box in enumerate(ndi.find_objects(plateaus), 1):
            if box is None: continue
            coords = np.argwhere(peaks[box]) + np.array([s.start for s in box])
            # Distinct plateau labels may share bounding boxes.
            labels_here = plateaus[tuple(coords.T)]
            coords = coords[labels_here == label_id]
            points.append(tuple(coords[0]))
        points = sorted(set(points), key=lambda p: (-score[p], p))
        accepted = []
        for point in points:
            if all(np.linalg.norm((np.array(point)-p)*spacing) >= cfg['separation_mm'] for p in accepted):
                accepted.append(point)
        support_labels, _ = ndi.label(liver & (score > cfg['low']), CONNECTIVITY)
        groups = {}
        for point in accepted:
            groups.setdefault(int(support_labels[point]), []).append(point)
        for label, seeds in sorted(groups.items()):
            if not label: continue
            support = support_labels == label
            coords = np.argwhere(support)
            margin = np.ceil(cfg['margin_mm']/spacing).astype(int)
            lo = np.maximum(coords.min(0)-margin, 0)
            hi = np.minimum(coords.max(0)+margin+1, liver.shape)
            box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            seed_mask = np.zeros(support[box].shape, bool)
            origins = []
            for point in seeds:
                seed_mask[tuple(np.array(point)-lo)] = True
                origins.append(dict(pixel=list(map(int, point)), score=float(score[point]),
                                    background_mm=cfg['background_mm'][prepared['scales'][polarity][point]]))
            candidates.append(Candidate(len(candidates)+1, polarity, box, seed_mask, support[box], origins))
    return dict(candidates=candidates, scores=scores, context_used=bool(cfg['context'] and neighbors),
                context_note='available' if cfg['context'] and neighbors else 'no_neighbors' if cfg['context'] else 'disabled')


def connected_to_seeds(mask, seeds):
    labels, count = ndi.label(mask, CONNECTIVITY)
    keep = np.zeros(count+1, bool); keep[np.unique(labels[seeds & mask])] = True; keep[0] = False
    return keep[labels]


def grow(image, liver, seeds, score, spacing, cfg):
    mask = seeds.copy(); iterations = 0
    while True:
        median, scale = robust(image[mask], cfg['noise_floor_hu'])
        border = ndi.binary_dilation(mask, CONNECTIVITY) & liver & ~mask
        add = border & (score > cfg['low']) & (np.abs(image-median) <= cfg['refinement']*scale)
        if not add.any(): break
        mask |= add; iterations += 1
    return mask, dict(iterations=iterations, converged=True)


def _signed_distance(mask, spacing):
    return ndi.distance_transform_edt(mask, sampling=spacing)-ndi.distance_transform_edt(~mask, sampling=spacing)


def _gradient(image, spacing):
    return [np.gradient(image, spacing[i], axis=i) if image.shape[i] > 1 else np.zeros_like(image)
            for i in range(2)]


def local_contour(image, liver, seeds, support, spacing, cfg):
    """Localized region fitting: each pixel has distinct interior/exterior means.

    Positive-inside level set with curvature regularization; reported fitting
    energy is diagnostic, not claimed to decrease monotonically under reinit.
    """
    if not (liver & ~support).any(): raise ValueError('missing_local_background')
    center, scale = robust(image[liver], cfg['noise_floor_hu'])
    data = (image-center)/scale
    kernel = physical_disk(cfg['window_mm'], spacing).astype(float)
    phi = _signed_distance(support, spacing)
    stable = 0; history = []; converged = False
    for iteration in range(1, cfg['max_iterations']+1):
        inside = (phi > 0) & liver; outside = liver & ~inside
        ni = ndi.convolve(inside.astype(float), kernel, mode='constant')
        no = ndi.convolve(outside.astype(float), kernel, mode='constant')
        ci = np.divide(ndi.convolve(data*inside, kernel, mode='constant'), ni, out=np.zeros_like(data), where=ni>0)
        co = np.divide(ndi.convolve(data*outside, kernel, mode='constant'), no, out=np.zeros_like(data), where=no>0)
        valid = (ni > 0) & (no > 0) & liver
        fit = np.where(valid, (data-co)**2-(data-ci)**2, 0.)
        gx, gy = _gradient(phi, spacing); norm = np.sqrt(gx*gx+gy*gy+1e-12)
        curvature = _gradient(gx/norm, spacing)[0]+_gradient(gy/norm, spacing)[1]
        speed = fit + cfg['refinement']*curvature
        band = liver & (np.abs(phi) <= 2*max(spacing))
        maximum = max(float(np.max(np.abs(speed[band]))) if band.any() else 0., 1e-12)
        phi[band] += .25*min(spacing)*speed[band]/maximum
        phi[seeds] = np.maximum(phi[seeds], min(spacing))
        phi[~liver] = -min(spacing)
        mask = (phi > 0) & liver
        changed = int(np.count_nonzero(mask != inside))/max(int(liver.sum()), 1)
        history.append(float(np.sum(np.where(mask, (data-ci)**2, (data-co)**2)[valid])))
        stable = stable+1 if changed < .001 else 0
        if stable >= 5:
            converged = True; break
        if iteration % 10 == 0:
            phi = _signed_distance(mask, spacing)
    return mask, dict(iterations=iteration, converged=converged, fitting_energy=history,
                      stable_mask_iterations=stable)


def binary_graph_cut(cost_foreground, cost_background, edges, foreground, background, valid):
    """Integer s/t min-cut. edges=(pixel_i,pixel_j,nonnegative Potts cost).

    Quantize all finite terms with a common scale <=1024. Their integer sum
    stays below 2**28; hard terminals cost sum+1, hence cannot be cut by an
    optimum feasible cut. Return source-reachable pixels as foreground.
    """
    valid=np.asarray(valid,dtype=bool)
    foreground=np.asarray(foreground,dtype=bool);background=np.asarray(background,dtype=bool)
    shape = valid.shape; n = valid.size
    if valid.ndim!=2 or any(np.shape(x)!=shape for x in (cost_foreground,cost_background,foreground,background)):
        raise ValueError('graph_shape_mismatch')
    if (foreground & ~valid).any() or (background & ~valid).any():
        raise ValueError('graph_markers_outside_roi')
    if not foreground.any() or not background.any() or (foreground & background).any():
        raise ValueError('missing_or_conflicting_graph_markers')
    cf, cb = np.asarray(cost_foreground, float).ravel(), np.asarray(cost_background, float).ravel()
    a, b, w = (np.asarray(x) for x in edges)
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=w.shape or np.any(a<0) or np.any(b<0) or np.any(a>=n) or np.any(b>=n):
        raise ValueError('invalid_graph_edges')
    rows = np.concatenate((np.full(n,n), np.arange(n), a, b)).astype(int)
    cols = np.concatenate((np.arange(n), np.full(n,n+1), b, a)).astype(int)
    values = np.concatenate((cb, cf, w, w))
    if not np.isfinite(values).all() or (values < 0).any(): raise ValueError('invalid_graph_costs')
    positive = values > 0
    budget = 2**28
    scale = min(1024., (budget-len(values)-1)/max(float(values.sum()), 1.))
    if scale <= 0: raise ValueError('graph_capacity_budget_exceeded')
    capacities = np.where(positive, np.maximum(1, np.floor(values*scale)), 0).astype(np.int64)
    hard = int(capacities.sum())+1
    fg = foreground.ravel(); bg = (background | ~valid).ravel()
    capacities[:n][fg] = hard
    capacities[n:2*n][bg] = hard
    graph = coo_matrix((capacities, (rows,cols)), shape=(n+2,n+2)).tocsr()
    graph.sum_duplicates()
    if graph.data.max(initial=0) >= 2**31: raise ValueError('graph_capacity_overflow')
    flow = maximum_flow(graph, n, n+1, method='dinic')
    residual = graph-flow.flow
    residual.data = (residual.data > 0).astype(np.int8); residual.eliminate_zeros()
    reached = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    mask = np.zeros(n+2, bool); mask[reached] = True; mask = mask[:n].reshape(shape) & valid
    if not mask[foreground].all() or mask[background].any(): raise AssertionError('Graph marker constraint violated')
    return mask, dict(capacity_scale=scale, hard_capacity=hard, flow_value=int(flow.flow_value))


def graphcut(image, liver, seeds, support, spacing, cfg):
    background = liver & ~support
    mf, sf = robust(image[seeds], cfg['noise_floor_hu'])
    mb, sb = robust(image[background], cfg['noise_floor_hu'])
    cf = np.minimum(.5*((image-mf)/sf)**2, 25.)
    cb = np.minimum(.5*((image-mb)/sb)**2, 25.)
    ids = np.arange(image.size).reshape(image.shape)
    aa=[]; bb=[]; ww=[]
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        x=slice(0,image.shape[0]-dx); xx=slice(dx,None)
        y=slice(max(0,-dy),image.shape[1]-max(0,dy)); yy=slice(max(0,dy),image.shape[1]-max(0,-dy))
        first=(x,y); second=(xx,yy)
        valid=liver[first]&liver[second]
        distance=np.hypot(dx*spacing[0],dy*spacing[1])
        weight=cfg['refinement']*np.exp(-.5*((image[first]-image[second])/max(sf,sb))**2)/distance
        aa.extend(ids[first][valid]); bb.extend(ids[second][valid]); ww.extend(weight[valid])
    return binary_graph_cut(cf,cb,(aa,bb,ww),seeds,background,liver)


def refine_candidates(prepared, generated, settings):
    cfg=validate_settings(settings)
    image,liver,spacing=prepared['image'],prepared['liver'],prepared['spacing']
    final=np.zeros(liver.shape,bool); seed_map=final.copy(); support_map=final.copy(); rows=[]; masks=[]
    for candidate in generated['candidates']:
        box=candidate.box; seeds=candidate.seeds; support=candidate.support
        seed_map[box] |= seeds; support_map[box] |= support
        record=dict(candidate_id=candidate.identifier,polarity=candidate.polarity,origins=candidate.origins,
                    box=[[s.start,s.stop] for s in box],support_area_mm2=float(support.sum()*np.prod(spacing)))
        try:
            if cfg['method']=='local_growth':
                mask,diagnostics=grow(image[box],liver[box],seeds,generated['scores'][candidate.polarity][box],spacing,cfg)
            elif cfg['method']=='local_contour':
                mask,diagnostics=local_contour(image[box],liver[box],seeds,support,spacing,cfg)
            else:
                mask,diagnostics=graphcut(image[box],liver[box],seeds,support,spacing,cfg)
            mask=connected_to_seeds(mask & liver[box],seeds)
            boundary=np.zeros(mask.shape,bool); boundary[[0,-1],:]=True; boundary[:,[0,-1]]=True
            record.update(status='ok',area_mm2=float(mask.sum()*np.prod(spacing)),
                          touches_roi_boundary=bool((mask & boundary).any()),**diagnostics)
            final[box] |= mask
        except ValueError as exc:
            mask=np.zeros(support.shape,bool)
            record.update(status='failed',reason=str(exc),area_mm2=0.)
        rows.append(record); masks.append((box,mask))
    return dict(mask=final,seeds=seed_map,candidates=support_map,components=rows,
                candidate_masks=masks,scores=generated['scores'],context_note=generated['context_note'],
                effective_parameters=cfg,failed_candidates=sum(r['status']=='failed' for r in rows))


def detect_local(hu,liver,spacing,settings,*,context=None):
    cfg=validate_settings(settings)
    if cfg['context'] and context is None:
        raise ValueError('Este método requiere cortes vecinos explícitos; este flujo no está adaptado.')
    prepared=prepare_maps(hu,liver,spacing,cfg)
    neighbors=prepare_context_maps(context,prepared['image'].shape,spacing,cfg) if cfg['context'] else None
    return refine_candidates(prepared,build_candidates(prepared,cfg,neighbors),cfg)
