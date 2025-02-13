import pandas as pd

base_path = "data/labels"


def load_labels(
    csv_meniscus_filepath : str,
    csv_acl_filepath : str,
    csv_abnormal_filepath : str,
) -> dict :
    
    df_meniscus = pd.read_csv(csv_meniscus_filepath, dtype={0: str}, header=None)
    df_acl = pd.read_csv(csv_acl_filepath, dtype={0: str}, header=None)
    df_abnormal = pd.read_csv(csv_abnormal_filepath, dtype={0: str}, header=None)
    
    return {
        "df_meniscus" : df_meniscus,
        "df_acl" : df_acl,
        "df_abnormal" : df_abnormal,
    }
    
    

def transform_labels(
    df_meniscus : pd.DataFrame,
    df_acl : pd.DataFrame,
    df_abnormal : pd.DataFrame,
) -> pd.DataFrame :
    
    df_meniscus = df_meniscus.rename(columns={0 : "PatientID", 1: "Meniscus"})
    df_acl = df_acl.rename(columns={0 : "PatientID", 1: "ACL"})
    df_abnormal = df_abnormal.rename(columns={0 : "PatientID", 1: "Abnormal"})
    
    return df_meniscus.merge(df_acl, on="PatientID").merge(df_abnormal, on="PatientID")



def run():
    df_train = transform_labels(
        **load_labels(
            csv_meniscus_filepath = base_path + "/train-meniscus.csv",
            csv_acl_filepath = base_path + "/train-acl.csv",
            csv_abnormal_filepath = base_path + "/train-abnormal.csv",
        ),
    )

    df_val = transform_labels(
        **load_labels(
            csv_meniscus_filepath = base_path + "/valid-meniscus.csv",
            csv_acl_filepath = base_path + "/valid-acl.csv",
            csv_abnormal_filepath = base_path + "/valid-abnormal.csv",
        ),
    )
    
    df_full = pd.concat([df_train, df_val], ignore_index=True)
    df_full.to_csv(base_path + "/processed_labels.csv", index=False)
    

run()