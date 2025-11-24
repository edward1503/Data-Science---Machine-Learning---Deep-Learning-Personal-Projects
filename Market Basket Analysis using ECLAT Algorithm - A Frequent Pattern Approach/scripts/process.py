import pandas as pd

# Hàm chuyển đổi các dạng kết quả thành dataframe
def convert_to_dataframe(frequent_itemsets, transaction_count):
    data = [
        {
            "itemsets": set(itemset),
            "support": support,
            "frequency": round(support * transaction_count)
        }
        for itemset, support in frequent_itemsets.items()
    ]

    return pd.DataFrame(data)


def rules_to_dataframe(rules):
    return pd.DataFrame([
        {
            "Antecedent": rule["antecedent"],
            "Consequence": rule["consequence"],
            "Confidence": rule["confidence"],
            "Support": rule["support"],
            "Lift": rule["lift"]
        }
        for rule in rules
    ])