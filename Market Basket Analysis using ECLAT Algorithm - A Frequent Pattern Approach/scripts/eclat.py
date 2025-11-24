from collections import defaultdict


class Eclat:
    """
    ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal) Algorithm.

    The ECLAT algorithm is used to find frequent itemsets in transactional data
    using a vertical representation (item-to-transaction mapping). It also supports
    the generation of association rules.

    Parameters:
        min_support (float): The minimum support threshold for itemsets (default: 0.01).
        verbose (bool): If True, print progress and details during execution (default: False).

    Attributes:
        itemsets (list): List of frequent itemsets with their transaction IDs.
        item_support (dict): Support values for each frequent itemset.
        transaction_count (int): Total number of transactions.
    """

    def __init__(self, min_support=0.01, verbose=False):
        self.min_support = min_support
        self.verbose = verbose
        self.itemsets = []
        self.item_support = {}
        self.transaction_count = 0

    def fit(self, transactions):
        """
        Fit the model to transactional data.

        Args:
            transactions (list of list): A list of transactions, where each transaction
                                         is a list of items.

        Returns:
            self: The fitted ECLAT model.
        """
        self.transaction_count = len(transactions)
        vertical_db = defaultdict(set)
        # Chuyển đổi dữ liệu thanh vertical database
        for tid, transaction in enumerate(transactions):
            unique_items = set(transaction)
            for item in unique_items:
                vertical_db[item].add(tid)

        if self.verbose:
            print(f"Initial vertical database size: {len(vertical_db)} items")
        # Tìm kiếm tập phổ biến
        self.itemsets = self._mine(vertical_db, min_length=1)
        # Tính support
        self.item_support = {
            itemset: len(tids) / self.transaction_count
            for itemset, tids in self.itemsets.items()
        }

        return self

    def _mine(self, vertical_db, prefix=set(), min_length=1):
        """
        Recursively mine frequent itemsets.

        Args:
            vertical_db (dict): Vertical database mapping items to transaction IDs.
            prefix (set): Current prefix of items.
            min_length (int): Minimum length of itemsets to consider.

        Returns:
            dict: Frequent itemsets with their transaction IDs.
        """
        frequent_itemsets = {}
        # Duyệt qua từng item
        for item, tids in vertical_db.items():
            new_itemset = prefix | {item}
            support = len(tids) / self.transaction_count

            if support >= self.min_support: # Kiểm tra thỏa minSup
                frequent_itemsets[frozenset(new_itemset)] = tids
                # Intersect các tID mới
                new_vertical_db = {
                    other_item: other_tids & tids
                    for other_item, other_tids in vertical_db.items()
                    if other_item > item
                }

                frequent_itemsets.update(self._mine(new_vertical_db, new_itemset, min_length))

        return frequent_itemsets

    def get_frequent_itemsets(self):
        """
        Get the frequent itemsets and their support values.

        Returns:
            dict: Frequent itemsets (frozenset) and their support values.
        """
        return {
            frozenset(itemset): support
            for itemset, support in self.item_support.items()
        }

    def generate_rules(self, min_confidence=0.5):
        """
        Generate association rules from frequent itemsets, including Lift.

        Args:
            min_confidence (float): Minimum confidence threshold for rules.

        Returns:
            list: List of association rules, each represented as a dictionary with keys:
                  'antecedent', 'consequence', 'confidence', 'support', and 'lift'.
        """
        rules = []
        for itemset in self.item_support.keys():
            if len(itemset) > 1:
                for consequence in itemset:
                    antecedent = itemset - {consequence}
                    confidence = self.item_support[itemset] / self.item_support[antecedent]

                    if confidence >= min_confidence:
                        lift = confidence / self.item_support[frozenset({consequence})]
                        rules.append({
                            "antecedent": set(antecedent),
                            "consequence": {consequence},
                            "confidence": confidence,
                            "support": self.item_support[itemset],
                            "lift": lift
                        })
        return rules