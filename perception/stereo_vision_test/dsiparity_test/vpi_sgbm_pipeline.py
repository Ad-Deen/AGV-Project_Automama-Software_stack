import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import cv2
  

def read_raw_file(fpath, resize_to=None, verbose=False):
    try:
        if verbose:
            print(f'I Reading: {fpath}', end=' ', flush=True)
        f = open(fpath, 'rb')
        np_arr = np.fromfile(f, dtype=np.uint16, count=-1)
        f.close()
        if verbose:
            print(f'done!\nI Raw array: shape: {np_arr.shape} dtype: {np_arr.dtype}')
        if resize_to is not None:
            np_arr = np_arr.reshape(resize_to, order='C')
        if verbose:
            print(f'I Reshaped array: shape: {np_arr.shape} dtype: {np_arr.dtype}')
        pil_img = Image.fromarray(np_arr, mode="I;16L")
        return pil_img
    except:
        raise ValueError(f'E Cannot process raw input: {fpath}')


def process_arguments():
    parser = ArgumentParser()

    parser.add_argument('backend', choices=['cuda','ofa','ofa-pva-vic'],
                        help='Backend to be used for processing')
    parser.add_argument('left', help='Rectified left input image from a stereo pair')
    parser.add_argument('right', help='Rectified right input image from a stereo pair')
    parser.add_argument('--width', default=-1, type=int, help='Input width for raw input files')
    parser.add_argument('--height', default=-1, type=int, help='Input height for raw input files')
    parser.add_argument('--downscale', default=1, type=int, help='Output downscale factor')
    parser.add_argument('--window_size', default=5, type=int, help='Median filter window size')
    parser.add_argument('--skip_confidence', default=False, action='store_true', help='Do not calculate confidence')
    parser.add_argument('--conf_threshold', default=32767, type=int, help='Confidence threshold')
    parser.add_argument('--conf_type', default='best', choices=['best', 'absolute', 'relative', 'inference'],
                        help='Computation type to produce the confidence output. Default will pick best option given backend.')
    parser.add_argument('-p1', default=3, type=int, help='Penalty P1 on small disparities')
    parser.add_argument('-p2', default=48, type=int, help='Penalty P2 on large disparities')
    parser.add_argument('--p2_alpha', default=0, type=int, help='Alpha for adaptive P2 Penalty')
    parser.add_argument('--uniqueness', default=-1, type=float, help='Uniqueness ratio')
    parser.add_argument('--skip_diagonal', default=False, action='store_true', help='Do not use diagonal paths')
    parser.add_argument('--num_passes', default=3, type=int, help='Number of passes')
    parser.add_argument('--min_disparity', default=0, type=int, help='Minimum disparity')
    parser.add_argument('--max_disparity', default=256, type=int, help='Maximum disparity')
    parser.add_argument('--output_mode', default=0, type=int, help='0: color; 1: grayscale; 2: raw binary')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose mode')

    return parser.parse_args()


def main():
    args = process_arguments()

    scale = 1 # pixel value scaling factor when loading input

    if args.backend == 'cuda':
        backend = vpi.Backend.CUDA
    elif args.backend == 'ofa':
        backend = vpi.Backend.OFA
    elif args.backend == 'ofa-pva-vic':
        backend = vpi.Backend.OFA|vpi.Backend.PVA|vpi.Backend.VIC
    else:
        raise ValueError(f'E Invalid backend: {args.backend}')

    conftype = None
    if args.conf_type == 'best':
        conftype = vpi.ConfidenceType.INFERENCE if args.backend == 'ofa-pva-vic' else vpi.ConfidenceType.ABSOLUTE
    elif args.conf_type == 'absolute':
        conftype = vpi.ConfidenceType.ABSOLUTE
    elif args.conf_type == 'relative':
        conftype = vpi.ConfidenceType.RELATIVE
    elif args.conf_type == 'inference':
        conftype = vpi.ConfidenceType.INFERENCE
    else:
        raise ValueError(f'E Invalid confidence type: {args.conf_type}')

    minDisparity = args.min_disparity
    maxDisparity = args.max_disparity
    includeDiagonals = not args.skip_diagonal
    numPasses = args.num_passes
    calcConf = not args.skip_confidence
    downscale = args.downscale
    windowSize = args.window_size
    quality = 6

    if args.verbose:
        print(f'I Backend: {backend}\nI Left image: {args.left}\nI Right image: {args.right}\n'
            f'I Disparities (min, max): {(minDisparity, maxDisparity)}\n'
            f'I Input scale factor: {scale}\nI Output downscale factor: {downscale}\n'
            f'I Window size: {windowSize}\nI Quality: {quality}\n'
            f'I Calculate confidence: {calcConf}\nI Confidence threshold: {args.conf_threshold}\n'
            f'I Confidence type: {conftype}\nI Uniqueness ratio: {args.uniqueness}\n'
            f'I Penalty P1: {args.p1}\nI Penalty P2: {args.p2}\nI Adaptive P2 alpha: {args.p2_alpha}\n'
            f'I Include diagonals: {includeDiagonals}\nI Number of passes: {numPasses}\n'
            f'I Output mode: {args.output_mode}\nI Verbose: {args.verbose}\n'
            , end='', flush=True)

    if 'raw' in args.left:
        pil_left = read_raw_file(args.left, resize_to=[args.height, args.width], verbose=args.verbose)
        np_left = np.asarray(pil_left)
    else:
        try:
            pil_left = Image.open(args.left)
            if pil_left.mode == 'I':
                np_left = np.asarray(pil_left).astype(np.int16)
            else:
                np_left = np.asarray(pil_left)
        except:
            raise ValueError(f'E Cannot open left input image: {args.left}')

    if 'raw' in args.right:
        pil_right = read_raw_file(args.right, resize_to=[args.height, args.width], verbose=args.verbose)
        np_right = np.asarray(pil_right)
    else:
        try:
            pil_right = Image.open(args.right)
            if pil_right.mode == 'I':
                np_right = np.asarray(pil_right).astype(np.int16)
            else:
                np_right = np.asarray(pil_right)
        except:
            raise ValueError(f'E Cannot open right input image: {args.right}')

    # Streams for left and right independent pre-processing
    streamLeft = vpi.Stream()
    streamRight = vpi.Stream()

    # Load input into a vpi.Image and convert it to grayscale, 16bpp
    with vpi.Backend.CUDA:
        with streamLeft:
            left = vpi.asimage(np_left).convert(vpi.Format.Y16_ER, scale=scale)
        with streamRight:
            right = vpi.asimage(np_right).convert(vpi.Format.Y16_ER, scale=scale)

    # Preprocess input
    # Block linear format is needed for ofa backends
    # We use VIC backend for the format conversion because it is low power
    if args.backend in {'ofa-pva-vic', 'ofa'}:
        if args.verbose:
            print(f'W {args.backend} forces to convert input images to block linear', flush=True)
        with vpi.Backend.VIC:
            with streamLeft:
                left = left.convert(vpi.Format.Y16_ER_BL)
            with streamRight:
                right = right.convert(vpi.Format.Y16_ER_BL)

    if args.verbose:
        print(f'I Input left image: {left.size} {left.format}\n'
            f'I Input right image: {right.size} {right.format}', flush=True)

    confidenceU16 = None

    if calcConf:
        if args.backend not in {'cuda', 'ofa-pva-vic'}:
            # Only CUDA and OFA-PVA-VIC support confidence map
            calcConf = False
            if args.verbose:
                print(f'W {args.backend} does not allow to calculate confidence', flush=True)


    outWidth = (left.size[0] + downscale - 1) // downscale
    outHeight = (left.size[1] + downscale - 1) // downscale

    if calcConf:
        confidenceU16 = vpi.Image((outWidth, outHeight), vpi.Format.U16)

    # Use stream left to consolidate actual stereo processing
    streamStereo = streamLeft

    if args.backend == 'ofa-pva-vic' and maxDisparity not in {128, 256}:
        maxDisparity = 128 if (maxDisparity // 128) < 1 else 256
        if args.verbose:
            print(f'W {args.backend} only supports 128 or 256 maxDisparity. Overriding to {maxDisparity}', flush=True)

    if args.verbose:
        if 'ofa' not in args.backend:
            print('W Ignoring P2 alpha and number of passes since not an OFA backend', flush=True)
        if args.backend != 'cuda':
            print('W Ignoring uniqueness since not a CUDA backend', flush=True)
        print('I Estimating stereo disparity ... ', end='', flush=True)

    # Estimate stereo disparity.
    with streamStereo, backend:
        disparityS16 = vpi.stereodisp(left, right, downscale=downscale, out_confmap=confidenceU16,
                                    window=windowSize, maxdisp=maxDisparity, confthreshold=args.conf_threshold,
                                    quality=quality, conftype=conftype, mindisp=minDisparity,
                                    p1=args.p1, p2=args.p2, p2alpha=args.p2_alpha, uniqueness=args.uniqueness,
                                    includediagonals=includeDiagonals, numpasses=numPasses)

    if args.verbose:
        print('done!\nI Post-processing ... ', end='', flush=True)

    # Postprocess results and save them to disk
    with streamStereo, vpi.Backend.CUDA:
        # Some backends outputs disparities in block-linear format, we must convert them to
        # pitch-linear for consistency with other backends.
        if disparityS16.format == vpi.Format.S16_BL:
            disparityS16 = disparityS16.convert(vpi.Format.S16, backend=vpi.Backend.VIC)

        # Scale disparity and confidence map so that values like between 0 and 255.

        # Disparities are in Q10.5 format, so to map it to float, it gets
        # divided by 32. Then the resulting disparity range, from 0 to
        # stereo.maxDisparity gets mapped to 0-255 for proper output.
        # Copy disparity values back to the CPU.
        disparityU8 = disparityS16.convert(vpi.Format.U8, scale=255.0/(32*maxDisparity)).cpu()

        # Apply JET colormap to turn the disparities into color, reddish hues
        # represent objects closer to the camera, blueish are farther away.
        disparityColor = cv2.applyColorMap(disparityU8, cv2.COLORMAP_JET)

        # Converts to RGB for output with PIL.
        disparityColor = cv2.cvtColor(disparityColor, cv2.COLOR_BGR2RGB)

        if calcConf:
            confidenceU8 = confidenceU16.convert(vpi.Format.U8, scale=255.0/65535).cpu()

            # When pixel confidence is 0, its color in the disparity is black.
            mask = cv2.threshold(confidenceU8, 1, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            disparityColor = cv2.bitwise_and(disparityColor, mask)

    fext = '.raw' if args.output_mode == 2 else '.png'

    disparity_fname = f'disparity_python{sys.version_info[0]}_{args.backend}' + fext
    confidence_fname = f'confidence_python{sys.version_info[0]}_{args.backend}' + fext

    if args.verbose:
        print(f'done!\nI Disparity output: {disparity_fname}', flush=True)
        if calcConf:
            print(f'I Confidence output: {confidence_fname}', flush=True)

    # Save results to disk.
    try:
        if args.output_mode == 0:
            Image.fromarray(disparityColor).save(disparity_fname)
            if args.verbose:
                print(f'I Output disparity image: {disparityColor.shape} '
                    f'{disparityColor.dtype}', flush=True)
        elif args.output_mode == 1:
            Image.fromarray(disparityU8).save(disparity_fname)
            if args.verbose:
                print(f'I Output disparity image: {disparityU8.shape} '
                    f'{disparityU8.dtype}', flush=True)
        elif args.output_mode == 2:
            disparityS16.cpu().tofile(disparity_fname)
            if args.verbose:
                print(f'I Output disparity image: {disparityS16.size} '
                    f'{disparityS16.format}', flush=True)

        if calcConf:
            if args.output_mode == 0 or args.output_mode == 1:
                Image.fromarray(confidenceU8).save(confidence_fname)
                if args.verbose:
                    print(f'I Output confidence image: {confidenceU8.shape} '
                        f'{confidenceU8.dtype}', flush=True)
            else:
                confidenceU16.cpu().tofile(confidence_fname)
                if args.verbose:
                    print(f'I Output confidence image: {confidenceU16.size} '
                        f'{confidenceU16.format}', flush=True)

    except:
        raise ValueError(f'E Cannot write outputs: {disparity_fname}, {confidence_fname}\n'
                        f'E Using output mode: {args.output_mode}')


if __name__ == '__main__':
    main()