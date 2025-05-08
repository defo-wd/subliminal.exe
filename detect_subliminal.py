import cv2
import os
from skimage.metrics import structural_similarity as ssim

# 2枚の画像間のSSIM（構造的類似度）を計算する関数
def compute_ssim(f1, f2):
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(g1, g2, full=True)
    return score

# 最も異常な箇所の前後フレームを検出・保存
def detect_anomalous_frame_pair(video_path, save_all_frames=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("動画ファイルを開けませんでした")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("最初のフレーム読み込みに失敗しました")
        return

    # Debug
    if save_all_frames:
        all_frames_dir = "all_frames"
        os.makedirs(all_frames_dir, exist_ok=True)
        cv2.imwrite(os.path.join(all_frames_dir, f"frame_0000.jpg"), prev_frame)

    min_ssim = 1.0
    min_prev = None
    min_next = None
    min_frame_index = 1

    frame_index = 1
    print("フレーム間のSSIMスコアを計算中...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        score = compute_ssim(prev_frame, frame)
        print(f"[{frame_index}] SSIM: {score:.4f}")

        if score < min_ssim:
            min_ssim = score
            min_prev = prev_frame.copy()
            min_next = frame.copy()
            min_frame_index = frame_index
        if save_all_frames:
            filename = os.path.join(all_frames_dir, f"frame_{frame_index:04d}.jpg")
            cv2.imwrite(filename, frame)

        prev_frame = frame
        frame_index += 1

    cap.release()

    if min_prev is not None and min_next is not None:
        before_path = f"サブリミナルなフレーム_{min_frame_index - 1}.jpg"
        after_path = f"サブリミナルじゃないフレーム_{min_frame_index}.jpg"

        cv2.imwrite(before_path, min_prev)
        cv2.imwrite(after_path, min_next)

        print("")
        print("サブリミナルフレームの前後を保存しました：")
        print(f"  前のフレーム : {before_path}（フレーム {min_frame_index - 1}）")
        print(f"  後のフレーム : {after_path}（フレーム {min_frame_index}）")
        print(f"  最小SSIM値   : {min_ssim:.4f}")
    else:
        print("異常フレームを検出できませんでした")

    print("処理が完了しました。")

# Debug用
detect_anomalous_frame_pair("source/sample2.mp4", save_all_frames=False)
