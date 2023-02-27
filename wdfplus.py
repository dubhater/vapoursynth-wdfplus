import vapoursynth as vs
import edi_rpow2 as edi

def WDFPlus(input, strength=6, tmax=10, tmin=7, gpu=True):
    core = vs.core

    if input.format.color_family not in [vs.YUV, vs.GRAY]:
        raise RuntimeError("WDFPlus: the clip must be YUV or GRAY.")

    if input.format.bits_per_sample > 16:
        raise RuntimeError("WDFPlus: the clip must have 8-16 bits.")

    if input.format.sample_type == vs.FLOAT:
        raise RuntimeError("WDFPlus: the clip must have integer samples.")

    src     = input
    w       = src.width
    h       = src.height
    pad     = core.resize.Point(clip=src, width=w+8, height=h+8, src_left=-4, src_top=-4, src_width=w+8, src_height=h+8)
    grey    = core.std.ShufflePlanes(clips=pad, planes=0, colorfamily=vs.GRAY)
    if gpu:
        up2 = edi.nnedi3cl_rpow2(clip=grey, rfactor=2, correct_shift=None)
    else:
        up2 = edi.znedi3_rpow2(clip=grey, rfactor=2, correct_shift=None)
    up4     = core.resize.Spline36(clip=up2, width=(w+8)*4, height=(h+8)*4, src_left=0.25, src_top=0.25)
    edge    = core.warp.ASobel(clip=grey, thresh=255).warp.ABlur()
    warp    = core.warp.AWarp(clip=up4, mask=edge, depth=strength)
    msk     = core.tedgemask.TEdgeMask(clip=grey, threshold=0)
    for i in range(5):
        msk = core.std.Maximum(clip=msk)
    blur_1_matrix = [1, 2, 1,
                     2, 4, 2,
                     1, 2, 1]
    msk     = core.std.Convolution(clip=msk, matrix=blur_1_matrix)
    owarp   = core.std.MaskedMerge(clipa=grey, clipb=warp, mask=msk)
    msk_tmax = core.tedgemask.TEdgeMask(clip=grey, threshold=tmax)
    msk_tmin = core.tedgemask.TEdgeMask(clip=grey, threshold=tmin)
    msk     = core.misc.Hysteresis(clipa=msk_tmax, clipb=msk_tmin)
    msk     = core.std.Convolution(clip=msk, matrix=blur_1_matrix)
    out     = core.std.MaskedMerge(clipa=owarp, clipb=pad, mask=msk).std.Crop(left=4, right=4, top=4, bottom=4)
    final   = core.std.CopyFrameProps(clip=out, prop_src=input)
    return final
