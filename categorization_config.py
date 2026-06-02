"""
Default categorization rules and subcategories for the expense tracker.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def get_categorization_rules() -> Dict[str, List[str]]:
    """Keyword rules for main categories (matched on description + original_description)."""
    return {
        "Food": [
            "ローソン", "ﾛ-ｿﾝ", "セブンイレブン", "セブン−イレブン", "seven", "ファミリーマート",
            "ﾌｱﾐﾘ", "family mart", "lawson", "コンビニ", "convenience", "スーパー", "supermarket",
            "焼肉", "yakiniku", "居酒屋", "izakaya", "磯丸", "酒場", "ﾋﾞｰﾝｽﾞ", "beans",
            "カラオケ", "karaoke", "ﾄﾞﾝｷ", "don quixote", "ｻﾝﾄﾞﾗ", "restaurant", "cafe",
            "coffee", "starbucks", "mcdonalds", "ﾄﾙﾊﾞ", "ﾎﾟﾂﾌﾟ", "ﾄｳｷﾕｳ", "ﾋﾕﾂﾃ", "ﾖﾄﾞｸﾞ",
            "ﾄｳｷﾖｳﾐﾀｲ", "ﾎﾟﾌﾟﾗ", "pop up", "ﾗｿﾞｰﾅ", "ﾎﾟﾌﾟﾗ", "ｶﾞﾝｿｽﾞｼ", "ﾄﾞｸﾞ", "wood",
            "ﾄﾞｶﾝ", "酒場", "ﾋﾟﾂﾂ", "ｼｰｼﾔ", "bar",
        ],
        "Social Life": [
            "飲み会", "drinking", "パーティー", "party", "懇親会", "歓迎会", "送別会",
            "カラオケ", "karaoke", "会食", "networking",
        ],
        "Subscriptions": [
            "icloud", "apple com bill", "apple music", "amazon prime", "ｱﾏｿﾞﾝﾌﾟﾗｲﾑ",
            "google one", "netflix", "spotflix", "spotify", "hulu", "disney",
            "u-next", "Ｕ−ＮＥＸＴ", "楽天モバイル", "rakuten mobile", "楽天Ｔｕｒｂｏ", "通信料",
            "cursor", "chatgpt", "openai", "subscription", "月額", "年額", "製品代",
        ],
        "Household": [
            "家賃", "rent", "光熱費", "utility", "楽天でんき", "楽天エナジー", "楽天ガス",
            "電気", "electric", "ガス", "gas", "水道", "water", "ニトリ", "nitori", "イケア",
            "ｲｹｱ", "ikea", "ホームセンター", "ﾄﾞﾛ-ﾝ", "ﾄﾞﾛ-ﾝｼﾞﾖｳ",
        ],
        "Transportation": [
            "電車", "train", "バス", "bus", "タクシー", "taxi", "地下鉄", "subway",
            "モバイルパス", "ﾓﾊﾞｲﾙﾊﾟｽ", "mobile pass", "ＥＴＣ", "etc", "駐車", "parking",
            "ﾀｲﾑｽﾞ", "times", "ﾊﾟ-ｷﾝｸﾞ", "ｼﾞﾄﾞｳﾊﾝﾊﾞｲｷ", "ﾅﾘﾀ", "gasoline", "ガソリン",
            "アポロ", "idemitsu", "イデミツ", "fuel",
        ],
        "Vacation": [
            "旅行", "travel", "ホテル", "hotel", "booking.com", "booking", "ブッキング",
            "ﾌﾞｯｷﾝｸﾞ", "airasia", "www.airasia", "marin hotel", "king power", "tax free",
            "flight", "新幹線", "shinkansen", "温泉", "onsen", "リゾート", "resort", "宿泊",
            "sasco", "利用国",
        ],
        "Shopping & Retail": [
            "amazon.co.jp", "amazon", "ｱﾏｿﾞﾝ", "amzn", "zara", "ｻﾞﾗ", "uniqlo", "ユニクロ",
            "ﾄﾞﾝｷﾎｰﾃ", "ﾎﾟﾂﾌﾟ", "ｶﾚﾂﾀ", "shopping", "retail",
        ],
        "Health": [
            "病院", "hospital", "クリニック", "clinic", "歯科", "dental", "薬局", "pharmacy",
            "薬", "medicine", "診察", "treatment",
        ],
        "Apparel": [
            "服", "clothing", "靴", "shoes", "ファッション", "fashion", "zara", "h&m", "gap",
            "nike", "adidas",
        ],
        "Bills & Fees": [
            "手数料", "年会費", "回収事務", "楽天ゴールドカード年会費",
        ],
        "Investments & Transfers": [
            "ｶﾌﾞｼｷ", "証券", "株式", "投資", "broker", "sbi", "楽天証券",
        ],
        "Self-development": [
            "本", "book", "雑誌", "magazine", "講座", "course", "セミナー", "seminar",
            "ワークショップ", "training", "learning",
        ],
    }


def get_subcategories() -> Dict[str, Dict[str, List[str]]]:
    return {
        "Food": {
            "Groceries": [
                "ローソン", "ﾛ-ｿﾝ", "セブン", "ファミリ", "ﾌｱﾐﾘ", "コンビニ", "ﾄﾞﾝｷ", "ｻﾝﾄﾞﾗ",
            ],
            "Dinner/Eating Out": ["焼肉", "居酒屋", "磯丸", "酒場", "ﾋﾞｰﾝｽﾞ", "yakiniku", "izakaya"],
            "Lunch/Eating Out": ["lunch", "カフェ", "coffee", "ランチ"],
            "Beverages A": ["スターバックス", "コーヒー", "starbucks"],
        },
        "Transportation": {
            "Mobile Pass": ["ﾓﾊﾞｲﾙﾊﾟｽ", "モバイルパス"],
            "ETC": ["ＥＴＣ", "etc"],
            "Parking": ["ﾀｲﾑｽﾞ", "times", "ﾊﾟ-ｷﾝｸﾞ"],
        },
        "Household": {
            "Utilities": ["楽天でんき", "楽天エナジー", "楽天ガス", "電気", "ガス", "光熱"],
            "Furniture": ["ニトリ", "nitori", "イケア", "ｲｹｱ", "ikea"],
        },
        "Vacation": {
            "Hotels": ["booking", "hotel", "ホテル", "marin hotel"],
            "Flights": ["airasia", "flight", "航空"],
        },
    }


# Category priority when multiple rules match (first wins)
CATEGORY_MATCH_ORDER: Tuple[str, ...] = (
    "Vacation",
    "Investments & Transfers",
    "Bills & Fees",
    "Household",
    "Transportation",
    "Subscriptions",
    "Health",
    "Apparel",
    "Shopping & Retail",
    "Food",
    "Social Life",
    "Self-development",
)
