import argparse
import cv2
import numpy as np
from vision import detect_utensil_ellipse, segment_food_in_utensil, estimate_area_and_volume


def parse_args():
    p = argparse.ArgumentParser(description="Waste Volume Estimator (Webcam)")
    p.add_argument('--cam', type=int, default=0, help='Camera index (default 0)')
    p.add_argument('--utensil', type=str, default='auto', choices=['auto', 'plate', 'bowl'],
                   help='Utensil type hint (improves heuristics)')
    p.add_argument('--diameter-mm', type=float, default=None,
                   help='Inner diameter of utensil opening, in mm (for scale)')
    p.add_argument('--assumed-food-height-mm', type=float, default=15.0,
                   help='Assumed average food height (mm)')
    p.add_argument('--min-radius', type=int, default=60, help='Min circle radius (px)')
    p.add_argument('--max-radius', type=int, default=400, help='Max circle radius (px)')
    p.add_argument('--debug', action='store_true', help='Enable debug prints/visuals')
    return p.parse_args()


def draw_ellipse_overlay(frame, ellipse, color=(0, 255, 255)):
    (cx, cy), (MA, ma), angle = ellipse
    cv2.ellipse(frame, (int(cx), int(cy)), (int(MA/2), int(ma/2)), angle, 0, 360, color, 2)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print('Error: Could not open camera', args.cam)
        return

    show_seg = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Warning: Camera frame not received')
            break

        display = frame.copy()

        ellipse = detect_utensil_ellipse(frame, utensil_hint=args.utensil,
                                         min_radius=args.min_radius, max_radius=args.max_radius,
                                         debug=args.debug)

        percent_fill = None
        est_volume_ml = None

        if ellipse is not None:
            draw_ellipse_overlay(display, ellipse, (0, 255, 255))
            seg_mask, debug_views = segment_food_in_utensil(frame, ellipse, debug=args.debug)

            if seg_mask is not None:
                # Visualization overlay
                if show_seg:
                    overlay = display.copy()
                    overlay[seg_mask > 0] = (0, 0, 255)
                    display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

                percent_fill, est_volume_ml = estimate_area_and_volume(
                    ellipse, seg_mask, args.diameter_mm, args.utensil, args.assumed_food_height_mm
                )

        # Text HUD
        h = 24
        y0 = 30
        info = []
        info.append(f"Utensil: {args.utensil}")
        if percent_fill is not None:
            info.append(f"Fill: {percent_fill:.1f}% (by area)")
        if est_volume_ml is not None:
            info.append(f"Approx Volume: {est_volume_ml:.0f} ml")
        if args.diameter_mm:
            info.append(f"Scale: {args.diameter_mm} mm diameter")
        if show_seg:
            info.append("Seg: ON (press s)")
        else:
            info.append("Seg: OFF (press s)")

        for i, txt in enumerate(info):
            cv2.putText(display, txt, (10, y0 + i*h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('Waste Volume Estimator', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            show_seg = not show_seg

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
