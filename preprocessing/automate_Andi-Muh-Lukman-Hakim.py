import pandas as pd
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Mendapatkan path absolut direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_clean_data(file_path, target_col):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Menambahkan inplace=True agar kolom benar-benar terhapus
    if "Student_ID" in df.columns:
        df.drop(columns=["Student_ID"], inplace=True)

    target_mapping = {"Low": 0, "Medium": 1, "High": 2}
    df[target_col] = df[target_col].map(target_mapping)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def preprocess_and_save(X, y, output_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    # Fit dan Transform pada Data Latih (X_train)
    X_train_processed = preprocessor.fit_transform(X_train)

    # HANYA Transform pada Data Uji (X_test) untuk mencegah leakage
    X_test_processed = preprocessor.transform(X_test)

    # Ambil nama kolom baru hasil OneHotEncoding
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        cat_cols
    )
    all_feature_names = num_cols + cat_feature_names.tolist()

    # Konversi array numpy menjadi DataFrame yang rapi
    X_train_clean = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_clean = pd.DataFrame(X_test_processed, columns=all_feature_names)

    # Hardcode nama folder dihapus agar menggunakan parameter output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Gabungkan kembali X dan y
    train_data_clean = pd.concat(
        [X_train_clean, y_train.reset_index(drop=True)], axis=1
    )
    test_data_clean = pd.concat([X_test_clean, y_test.reset_index(drop=True)], axis=1)

    train_data_clean.to_csv(
        os.path.join(output_dir, "train_processed.csv"), index=False
    )
    test_data_clean.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)

    print("Proses preprocessing selesai. Data berhasil disimpan di folder:", output_dir)


if __name__ == "__main__":
    file_name = "ai_student_impact_dataset (1).csv"
    target_col = "Burnout_Risk_Level"

    final_csv_path = None

    try:
        print("1. Mencoba mengunduh dataset dari Kagglehub...")
        kaggle_path = kagglehub.dataset_download("laveshjadon/ai-impact-on-students")
        potential_path = os.path.join(kaggle_path, file_name)

        if os.path.exists(potential_path):
            final_csv_path = potential_path
            print(f"Berhasil menemukan file di: {final_csv_path}")
        else:
            raise FileNotFoundError(
                f"File '{file_name}' tidak ditemukan di {kaggle_path}"
            )

    except Exception as e:
        print(f"Gagal memuat dari Kagglehub ({e}).")

        # Fallback ke parent directory menggunakan SCRIPT_DIR
        fallback_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", file_name))
        fallback_path_alt = os.path.abspath(
            os.path.join(SCRIPT_DIR, "..", "ai_student_impact_dataset (1).csv")
        )

        print(f"2. Menggunakan fallback, mencari file di parent directory...")

        if os.path.exists(fallback_path):
            final_csv_path = fallback_path
            print(f"Ditemukan: {final_csv_path}")
        elif os.path.exists(fallback_path_alt):
            final_csv_path = fallback_path_alt
            print(f"Ditemukan file alternatif: {final_csv_path}")
        else:
            print(
                "ERROR: File CSV tidak ditemukan di Kagglehub maupun parent directory."
            )
            exit()

    print("\n3. Memulai pemrosesan data...")
    X, y = load_and_clean_data(final_csv_path, target_col)

    # Menentukan path output agar persis satu folder dengan file script
    output_folder_path = os.path.join(SCRIPT_DIR, "ai_impact_preprocessing")

    preprocess_and_save(X, y, output_folder_path)
