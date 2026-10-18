"""End-to-end prediction: CSV in (entity, chunk_text) -> CSV out (one ICD-10-CM code + one SNOMED CT concept per row).

  python predict.py --input my_terms.csv --output predictions.csv          # GPU if available
  python predict.py --input my_terms.csv --output predictions.csv --no-llm # skip the Qwen2.5 SNOMED gate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from termmap_plus.predict_sf import main  # noqa: E402

if __name__ == "__main__":
    main()
