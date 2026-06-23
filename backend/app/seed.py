"""Seed dataset.

Mirrors the seven detection events the NyayaChakshu console renders, plus the
analytics tables (per-class accuracy, junction load, today's class counts).
Serving this from the API lets the static console be backed by a real service
without touching its markup.
"""
from __future__ import annotations

# Console-shape cases (kept 1:1 with the frontend so responses drop in cleanly).
CASES: list[dict] = [
    {"id": "EVT-7741", "n": "01", "img": "triple", "cam": "CAM-07A",
     "loc": "CAM-07A · NH-65 service rd · NB", "plate": "AP 37 BK 6798", "pconf": 0.94,
     "frame": "#0241·887", "tracks": "6", "inf": "36 ms", "hash": "a91f…7c4e",
     "sig": "14:22:09", "gps": "17.4019°N, 78.4862°E",
     "boxes": [
         {"x": 13, "y": 34, "w": 18, "h": 41, "c": "red", "l": "rider · no helmet 0.96"},
         {"x": 14, "y": 62, "w": 13, "h": 7, "c": "amber", "l": "PLATE 0.88"},
         {"x": 37, "y": 26, "w": 18, "h": 48, "c": "red", "l": "triple-riding 0.90"},
         {"x": 39, "y": 64, "w": 12, "h": 7, "c": "amber", "l": "PLATE 6798 0.94"},
         {"x": 72, "y": 30, "w": 20, "h": 44, "c": "red", "l": "2 riders · no helmet 0.93"}],
     "viols": [
         {"nm": "Triple-riding", "c": "red", "sc": "0.90", "sec": "MV Act §194C", "fine": 1000, "code": "triple_riding", "riders": 1},
         {"nm": "Helmet absence — 3 riders", "c": "red", "sc": "0.96", "sec": "MV Act §194D", "fine": 3000, "code": "helmet_absence", "riders": 3}]},

    {"id": "EVT-7742", "n": "02", "img": "seatbelt", "cam": "CAM-07C",
     "loc": "CAM-07C · windshield ANPR lane", "plate": "PLATE OCCLUDED", "pconf": 0,
     "frame": "#0241·902", "tracks": "1", "inf": "39 ms", "hash": "6b30…1faa",
     "sig": "14:22:41", "gps": "17.4021°N, 78.4859°E",
     "boxes": [
         {"x": 22, "y": 15, "w": 55, "h": 50, "c": "amber", "l": "windshield ROI"},
         {"x": 33, "y": 27, "w": 27, "h": 33, "c": "red", "l": "seatbelt ABSENT 0.92"},
         {"x": 46, "y": 43, "w": 13, "h": 13, "c": "red", "l": "phone use 0.81"},
         {"x": 30, "y": 30, "w": 9, "h": 9, "c": "cyan", "l": "steering grip"}],
     "viols": [
         {"nm": "Seatbelt non-compliance (driver)", "c": "red", "sc": "0.92", "sec": "MV Act §194B", "fine": 1000, "code": "seatbelt_driver", "riders": 1},
         {"nm": "Handheld phone use", "c": "red", "sc": "0.81", "sec": "MV Act §184", "fine": 5000, "code": "phone_use", "riders": 1}]},

    {"id": "EVT-7743", "n": "03", "img": "helmet", "cam": "CAM-07B",
     "loc": "CAM-07B · overhead gantry · SB", "plate": "TS 09 EM 4521", "pconf": 0.79,
     "frame": "#0241·915", "tracks": "5", "inf": "35 ms", "hash": "d7c2…9b08",
     "sig": "14:23:02", "gps": "17.4017°N, 78.4865°E",
     "boxes": [
         {"x": 5, "y": 18, "w": 30, "h": 34, "c": "cyan", "l": "vehicle · tracked"},
         {"x": 40, "y": 30, "w": 13, "h": 24, "c": "red", "l": "no helmet 0.95"},
         {"x": 13, "y": 40, "w": 20, "h": 18, "c": "red", "l": "2 riders · no helmet 0.91"},
         {"x": 60, "y": 42, "w": 17, "h": 16, "c": "red", "l": "no helmet 0.94"},
         {"x": 66, "y": 8, "w": 13, "h": 18, "c": "red", "l": "no helmet 0.89"}],
     "viols": [
         {"nm": "Helmet absence — 5 riders", "c": "red", "sc": "0.95", "sec": "MV Act §194D", "fine": 5000, "code": "helmet_absence", "riders": 5}]},

    {"id": "EVT-7744", "n": "04", "img": "redlight", "cam": "CAM-07D",
     "loc": "CAM-07D · Paradise Jn · signal 3", "plate": "TS 07 GA 1180", "pconf": 0.91,
     "frame": "#0241·931", "tracks": "9", "inf": "38 ms", "hash": "3e57…aa20",
     "sig": "14:23:28", "gps": "17.4441°N, 78.4882°E",
     "boxes": [
         {"x": 46, "y": 2, "w": 6, "h": 9, "c": "red", "l": "SIGNAL: RED"},
         {"x": 40, "y": 24, "w": 25, "h": 18, "c": "red", "l": "red-light run 0.97"},
         {"x": 12, "y": 38, "w": 54, "h": 5, "c": "amber", "l": "STOP LINE"},
         {"x": 52, "y": 30, "w": 9, "h": 5, "c": "amber", "l": "PLATE 0.91"}],
     "viols": [
         {"nm": "Red-light running", "c": "red", "sc": "0.97", "sec": "MV Act §177 r/w §119", "fine": 5000, "code": "red_light", "riders": 1},
         {"nm": "Stop-line crossing", "c": "red", "sc": "0.92", "sec": "MV Act §177", "fine": 500, "code": "stop_line", "riders": 1}]},

    {"id": "EVT-7745", "n": "05", "img": "wrongside", "cam": "CAM-11A",
     "loc": "CAM-11A · Tank Bund flyover", "plate": "AP 28 CJ 7042", "pconf": 0.88,
     "frame": "#0188·402", "tracks": "14", "inf": "40 ms", "hash": "b2d9…4c61",
     "sig": "14:24:05", "gps": "17.4239°N, 78.4738°E",
     "boxes": [
         {"x": 43, "y": 24, "w": 13, "h": 20, "c": "red", "l": "wrong-side 0.96"},
         {"x": 42, "y": 42, "w": 13, "h": 8, "c": "amber", "l": "LANE DIR"},
         {"x": 45, "y": 40, "w": 8, "h": 4, "c": "amber", "l": "PLATE 0.88"},
         {"x": 60, "y": 18, "w": 14, "h": 9, "c": "cyan", "l": "track #14"}],
     "viols": [
         {"nm": "Wrong-side driving", "c": "red", "sc": "0.96", "sec": "MV Act §184", "fine": 5000, "code": "wrong_side", "riders": 1}]},

    {"id": "EVT-7746", "n": "06", "img": "parking", "cam": "CAM-04",
     "loc": "CAM-04 · Sri Krishna Mkt rd", "plate": "TS 08 FK 2299", "pconf": 0.86,
     "frame": "#0094·771", "tracks": "3", "inf": "34 ms", "hash": "7f10…d3b8",
     "sig": "14:24:50", "gps": "17.3981°N, 78.4901°E",
     "boxes": [
         {"x": 39, "y": 27, "w": 24, "h": 19, "c": "red", "l": "illegal parking 0.93"},
         {"x": 31, "y": 13, "w": 8, "h": 10, "c": "amber", "l": "NO-PARK ZONE"},
         {"x": 44, "y": 40, "w": 9, "h": 5, "c": "amber", "l": "PLATE 0.86"}],
     "viols": [
         {"nm": "Illegal parking", "c": "red", "sc": "0.93", "sec": "MV Act §15 r/w §127", "fine": 500, "code": "illegal_parking", "riders": 1}]},

    {"id": "EVT-7747", "n": "07", "img": "stopline", "cam": "CAM-07",
     "loc": "CAM-07 · South St signal", "plate": "TS 10 BH 6634", "pconf": 0.90,
     "frame": "#0241·948", "tracks": "8", "inf": "37 ms", "hash": "c4a1…22ef",
     "sig": "14:25:33", "gps": "17.4002°N, 78.4870°E",
     "boxes": [
         {"x": 38, "y": 22, "w": 26, "h": 24, "c": "red", "l": "stop-line cross 0.95"},
         {"x": 16, "y": 36, "w": 52, "h": 9, "c": "amber", "l": "ZEBRA / STOP LINE"},
         {"x": 80, "y": 6, "w": 6, "h": 9, "c": "red", "l": "SIGNAL: RED"},
         {"x": 50, "y": 34, "w": 9, "h": 5, "c": "amber", "l": "PLATE 0.90"}],
     "viols": [
         {"nm": "Stop-line crossing", "c": "red", "sc": "0.95", "sec": "MV Act §177", "fine": 500, "code": "stop_line", "riders": 1}]},
]

# [violation, precision, recall, F1, false-positive rate]
METRICS = [
    ["Helmet absence", 0.95, 0.91, 0.93, "1.8%"],
    ["Triple-riding", 0.90, 0.84, 0.87, "3.9%"],
    ["Seatbelt (driver)", 0.92, 0.84, 0.88, "2.4%"],
    ["Seatbelt (passenger)", 0.86, 0.74, 0.80, "4.1%"],
    ["Wrong-side driving", 0.96, 0.88, 0.92, "0.9%"],
    ["Stop-line crossing", 0.95, 0.90, 0.92, "1.2%"],
    ["Red-light running", 0.97, 0.89, 0.93, "0.8%"],
    ["Illegal parking", 0.93, 0.86, 0.89, "1.5%"],
]

# [label, count_today, colour, case_index]
CLASS_TODAY = [
    ["Helmet absence", 312, "#FF4D6D", 2],
    ["Triple-riding", 148, "#F6A609", 0],
    ["Seatbelt / phone", 97, "#2FD3C3", 1],
    ["Red-light running", 84, "#7B6CFF", 3],
    ["Wrong-side", 61, "#A99CFF", 4],
    ["Stop-line", 53, "#F25C9A", 6],
    ["Illegal parking", 41, "#35D199", 5],
]

# [name, camera, count, case_index]
JUNCTIONS = [
    ["Paradise Junction", "CAM-07D", 214, 3],
    ["Tank Bund flyover", "CAM-11A", 176, 4],
    ["Punjagutta X-rd", "CAM-09C", 141, 0],
    ["Sri Krishna Mkt", "CAM-04", 118, 5],
    ["South St signal", "CAM-07", 96, 6],
]

GAUGES = [
    {"pct": 93, "color": "#7B6CFF", "gl": "Detection accuracy", "gn": "93.0%", "gs": "▲ 0.4 mean F1"},
    {"pct": 71, "color": "#2FD3C3", "gl": "Auto-clear rate", "gn": "71%", "gs": "612 / 796 today"},
    {"pct": 100, "color": "#35D199", "gl": "Cameras online", "gn": "12/12", "gs": "all nodes nominal"},
    {"pct": 92, "color": "#F6A609", "gl": "Edge headroom", "gn": "37ms", "gs": "budget ≤ 40 ms"},
    {"pct": 78, "color": "#F25C9A", "gl": "Recovery vs target", "gn": "₹9.4L", "gs": "78% of daily target"},
]

REVIEW_DONUT = [
    {"c": "#35D199", "v": 612, "n": "Auto-cleared & issued"},
    {"c": "#F6A609", "v": 7, "n": "Pending review"},
    {"c": "#2FD3C3", "v": 142, "n": "Sent to senior"},
    {"c": "#FF4D6D", "v": 35, "n": "Dismissed (FP)"},
]
