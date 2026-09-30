import argparse
import torch
import evaluate
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from normalizer import normalize_vietnamese_text  # Hàm đã viết ở Ngày 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="JRHuy/vivos-fleurs", help="Tên dataset trên Hugging Face")
    parser.add_argument("--split", type=str, default="test", help="Tập dữ liệu dùng để test")
    parser.add_argument("--n", type=int, default=30, help="Số lượng mẫu chạy thử")
    args = parser.parse_args()

    # 1. Kiểm tra thiết bị tính toán (GPU / CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Đang chạy trên thiết bị: {device.upper()} ===")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Tải mô hình Whisper Small gốc và Processor
    model_id = "openai/whisper-small"
    print(f"\n[1/4] Đang tải mô hình gốc: {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    # 3. Tải n mẫu đầu tiên từ tập test
    print(f"\n[2/4] Đang tải {args.n} mẫu từ {args.dataset} (split: {args.split})...")
    ds = load_dataset(args.dataset, split=f"{args.split}[:{args.n}]")

    # 4. Khởi tạo metric đo lường
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions = []
    references = []

    print(f"\n[3/4] Bắt đầu suy luận (Inference)...")
    # Cấu hình giải mã tiếng Việt cho Whisper
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="vietnamese", task="transcribe")

    for i, item in enumerate(ds):
        audio = item["audio"]
        
        # Tiền xử lý dữ liệu âm thanh
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        # Giải mã ra văn bản
        transcription_raw = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        reference_raw = item.get("sentence", item.get("transcription", ""))

        # Chuẩn hóa văn bản trước khi tính điểm WER/CER
        pred_norm = normalize_vietnamese_text(transcription_raw)
        ref_norm = normalize_vietnamese_text(reference_raw)

        predictions.append(pred_norm)
        references.append(ref_norm)

        print(f"\n--- Mẫu {i+1}/{args.n} ---")
        print(f"Gốc (True): {ref_norm}")
        print(f"AI (Pred):  {pred_norm}")

    # 5. Tính toán kết quả WER và CER
    print(f"\n[4/4] Tính toán chỉ số đánh giá...")
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print("\n" + "="*50)
    print("KẾT QUẢ BASELINE (Mô hình gốc Whisper Small):")
    print(f" - Số mẫu đánh giá: {args.n}")
    print(f" - WER (Word Error Rate):      {wer * 100:.2f}%")
    print(f" - CER (Character Error Rate): {cer * 100:.2f}%")
    if device == "cuda":
        max_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f" - VRAM sử dụng tối đa:        {max_vram:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    main()