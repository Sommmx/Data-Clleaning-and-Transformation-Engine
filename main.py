import argparse
from src.cleaner import handling_missing_values, handling_outliers, normalize_numeric, encode_categorical
from src.utils import load_csv, load_header, save_csv

def main(input_file, output_file):
    header = load_header(input_file)
    data = load_csv(input_file)

    print("✅ Handling missing values...")
    data = handling_missing_values(data)

    print("------------------------------------------------------------")
    print(data)
    print("------------------------------------------------------------")

    print("✅ Handling outliers...")
    data = handling_outliers(data, threshold=3)

    print("✅ Normalizing numeric columns...")
    data = normalize_numeric(data, model='z-score')

    print("✅ Encoding categorical columns...")
    data, mapping = encode_categorical(data, save_mapping=True)

    print("✅ Saving cleaned data...")
    save_csv(data, header, output_file)
    print(f"🎉 Cleaning complete! Output saved to: {output_file}")
    print(f"Mapping used for categorical columns: {mapping}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production-level Data Cleaning Engine")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to save cleaned CSV file")
    args = parser.parse_args()
    main(args.input, args.output)
