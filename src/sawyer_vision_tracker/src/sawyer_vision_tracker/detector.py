"""Object detection via HSV color segmentation and ArUco markers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """A single detected object in a frame."""

    centroid: tuple[int, int]              # (cx, cy) in pixels
    bbox: tuple[int, int, int, int]        # (x, y, w, h)
    contour: np.ndarray
    area: float
    color_name: str
    color_bgr: tuple[int, int, int]
    mask: np.ndarray                       # binary mask for this object


class Detector:
    """Detect objects using HSV color segmentation and/or ArUco markers."""

    def __init__(self, config: dict) -> None:
        self._hsv_ranges = config.get("hsv_ranges", [])
        self._min_area = config.get("min_contour_area", 500)
        self._blur_k = config.get("gaussian_blur", 5)
        morph_k = config.get("morph_kernel_size", 5)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_k, morph_k)
        )

        # ArUco setup
        aruco_cfg = config.get("aruco", {})
        self._aruco_enabled = aruco_cfg.get("enabled", False)
        if self._aruco_enabled:
            dict_name = aruco_cfg.get("dictionary", "DICT_4X4_50")
            dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
            self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            self._aruco_params = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(
                self._aruco_dict, self._aruco_params
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect objects via HSV color segmentation.
        Returns list of Detection for each found object.
        """
        h, w = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (self._blur_k, self._blur_k), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        detections: list[Detection] = []

        for color_cfg in self._hsv_ranges:
            name = color_cfg["name"]
            color_bgr = tuple(color_cfg["color_bgr"])

            lower = np.array(color_cfg["lower"], dtype=np.uint8)
            upper = np.array(color_cfg["upper"], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            # handle red hue wrap-around with second range
            if "lower2" in color_cfg and "upper2" in color_cfg:
                lower2 = np.array(color_cfg["lower2"], dtype=np.uint8)
                upper2 = np.array(color_cfg["upper2"], dtype=np.uint8)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask, mask2)

            # morphological cleanup
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self._min_area:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                x, y, bw, bh = cv2.boundingRect(cnt)

                # per-object binary mask
                obj_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(obj_mask, [cnt], -1, 255, cv2.FILLED)

                detections.append(
                    Detection(
                        centroid=(cx, cy),
                        bbox=(x, y, bw, bh),
                        contour=cnt,
                        area=area,
                        color_name=name,
                        color_bgr=color_bgr,
                        mask=obj_mask,
                    )
                )

        return detections

    def detect_aruco(self, frame: np.ndarray) -> list[Detection]:
        """Detect ArUco markers and return as Detection objects."""
        if not self._aruco_enabled:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)

        if ids is None:
            return []

        h, w = frame.shape[:2]
        detections: list[Detection] = []

        for i, marker_corners in enumerate(corners):
            pts = marker_corners[0].astype(np.int32)
            marker_id = int(ids[i][0])

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            x, y, bw, bh = cv2.boundingRect(pts)
            area = float(cv2.contourArea(pts))

            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obj_mask, [pts], 255)

            detections.append(
                Detection(
                    centroid=(cx, cy),
                    bbox=(x, y, bw, bh),
                    contour=pts.reshape(-1, 1, 2),
                    area=area,
                    color_name=f"aruco_{marker_id}",
                    color_bgr=(0, 255, 255),
                    mask=obj_mask,
                )
            )

        return detections

    @staticmethod
    def get_combined_mask(
        detections: list[Detection], shape: tuple[int, int]
    ) -> np.ndarray:
        """OR all detection masks into a single binary mask."""
        combined = np.zeros(shape, dtype=np.uint8)
        for det in detections:
            combined = cv2.bitwise_or(combined, det.mask)
        return combined
