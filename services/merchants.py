"""
services/merchants.py — Static merchant library for zero-cost Japanese→English lookup.

This module provides a curated mapping of common Japanese merchant names (and their
half-width / mixed-script credit-card statement variants) to clean English display names.

How it fits into the translation pipeline
─────────────────────────────────────────
translate_batch_ai() checks sources in this order, cheapest first:

  Step 0 │ MERCHANT_LIBRARY  (this module, in-memory, ~0 μs, zero cost)
  Step 1 │ SQLite translation_cache  (DB lookup, ~1 ms, zero cost)
  Step 2 │ AI provider call  (network, ~1–5 s, may cost tokens)

A text that matches Step 0 never reaches Step 1 or 2.

Extending the library
─────────────────────
Add entries to the relevant section below.  The key must be the normalized
full-width form you would get after applying normalize_japanese_text() to the
raw statement text (NFKC + katakana hyphen fix).  Half-width katakana on
statements (ｽﾀｰﾊﾞｯｸｽ) is automatically converted to full-width (スターバックス)
by normalize_japanese_text() before the lookup.

If you see a merchant that the AI translated correctly but is not in this
library, add it here so future uploads are instant.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Core library — normalized full-width JP keys → English display names
# ─────────────────────────────────────────────────────────────────────────────

MERCHANT_LIBRARY: dict[str, str] = {

    # ── Convenience stores ────────────────────────────────────────────────────
    "ローソン": "Lawson",
    "ローソンストア100": "Lawson Store 100",
    "セブンイレブン": "7-Eleven",
    "7-ELEVEN": "7-Eleven",
    "7-11": "7-Eleven",
    "ファミリーマート": "FamilyMart",
    "ミニストップ": "Ministop",
    "デイリーヤマザキ": "Daily Yamazaki",
    "ポプラ": "Poplar",
    "ニューデイズ": "NewDays",
    "セイコーマート": "Seicomart",
    "スリーエフ": "Three F",

    # ── Supermarkets ──────────────────────────────────────────────────────────
    "イオン": "AEON",
    "イオンモール": "AEON Mall",
    "イオンスーパー": "AEON Supermarket",
    "マックスバリュ": "MaxValu",
    "ミニストップ": "Ministop",
    "イトーヨーカドー": "Ito-Yokado",
    "西友": "Seiyu",
    "ライフ": "LIFE Supermarket",
    "マルエツ": "Maruetsu",
    "サミット": "Summit",
    "オーケー": "OK Store",
    "オーケーストア": "OK Store",
    "コストコ": "Costco",
    "業務スーパー": "Gyomu Super",
    "ベルク": "Belc",
    "ヤオコー": "Yaoko",
    "東急ストア": "Tokyu Store",
    "成城石井": "Seijo Ishii",
    "紀ノ国屋": "Kinokuniya Supermarket",
    "ザ・ガーデン自由が丘": "The Garden Jiyugaoka",
    "クイーンズ伊勢丹": "Queens Isetan",
    "フードウェイ": "Foodway",

    # ── Fast food / casual dining ─────────────────────────────────────────────
    "マクドナルド": "McDonald's",
    "モスバーガー": "MOS Burger",
    "ケンタッキーフライドチキン": "KFC",
    "ケンタッキー": "KFC",
    "KFC": "KFC",
    "吉野家": "Yoshinoya",
    "すき家": "Sukiya",
    "松屋": "Matsuya",
    "なか卯": "Nakau",
    "てんや": "Tenya",
    "サイゼリヤ": "Saizeriya",
    "ガスト": "Gusto",
    "ジョナサン": "Jonathan's",
    "デニーズ": "Denny's",
    "バーミヤン": "Bamiyan",
    "夢庵": "Yumean",
    "ビッグボーイ": "Big Boy",
    "ロイヤルホスト": "Royal Host",
    "ファミレス": "Family Restaurant",
    "ここ壱": "CoCo Ichibanya",
    "カレーハウスCoCo壱番屋": "CoCo Ichibanya",
    "ドミノピザ": "Domino's Pizza",
    "ピザハット": "Pizza Hut",
    "ピザーラ": "Pizza-La",
    "サブウェイ": "Subway",
    "フレッシュネスバーガー": "Freshness Burger",
    "バーガーキング": "Burger King",
    "ロッテリア": "Lotteria",
    "シェーキーズ": "Shakey's",
    "鳥貴族": "Torikizoku",
    "鳥貴": "Torikizoku",
    "焼き鳥": "Yakitori",
    "串カツ田中": "Kushikatsu Tanaka",
    "天丼てんや": "Tenya",

    # ── Coffee / cafes ────────────────────────────────────────────────────────
    "スターバックス": "Starbucks",
    "スタバ": "Starbucks",
    "タリーズコーヒー": "Tully's Coffee",
    "タリーズ": "Tully's Coffee",
    "ドトールコーヒー": "Doutor Coffee",
    "ドトール": "Doutor Coffee",
    "コメダ珈琲": "Komeda Coffee",
    "コメダ": "Komeda Coffee",
    "サンマルクカフェ": "Saint Marc Café",
    "サンマルク": "Saint Marc Café",
    "プロント": "Pronto",
    "ベックスコーヒー": "BECK'S Coffee",
    "エクセルシオールカフェ": "Excelsior Café",
    "上島珈琲店": "Ueshima Coffee",
    "カフェ・ド・クリエ": "Café de Crié",
    "ブルーボトルコーヒー": "Blue Bottle Coffee",
    "スターバックスリザーブ": "Starbucks Reserve",
    "椿屋珈琲": "Tsubakiya Coffee",

    # ── Drugstores / pharmacy ─────────────────────────────────────────────────
    "マツモトキヨシ": "Matsumoto Kiyoshi",
    "ツルハドラッグ": "Tsuruha Drug",
    "ツルハ": "Tsuruha Drug",
    "ウエルシア薬局": "Welcia Pharmacy",
    "ウエルシア": "Welcia Pharmacy",
    "スギ薬局": "Sugi Pharmacy",
    "ドラッグストア": "Drugstore",
    "ここから": "Cocokara Fine",
    "ここからファイン": "Cocokara Fine",
    "アインファーマシーズ": "Ain Pharmacies",
    "大黒ドラッグ": "Daikoku Drug",
    "キリン堂": "Kirindo",
    "クスリのアオキ": "Kusuri no Aoki",
    "サンドラッグ": "Sun Drug",
    "ダイコクドラッグ": "Daikoku Drug",
    "ビー・アンド・ディー": "B&D Pharmacy",
    "日本調剤": "Nihon Chozai",
    "調剤薬局": "Pharmacy",

    # ── Electronics retail ────────────────────────────────────────────────────
    "ヨドバシカメラ": "Yodobashi Camera",
    "ヨドバシ": "Yodobashi Camera",
    "ビックカメラ": "Bic Camera",
    "ビック": "Bic Camera",
    "ヤマダ電機": "Yamada Denki",
    "ヤマダデンキ": "Yamada Denki",
    "ジョーシン": "Joshin",
    "上新電機": "Joshin",
    "ソフマップ": "Sofmap",
    "エディオン": "Edion",
    "コジマ": "Kojima",
    "ケーズデンキ": "K's Denki",

    # ── Fashion / apparel ─────────────────────────────────────────────────────
    "ユニクロ": "Uniqlo",
    "ジーユー": "GU",
    "しまむら": "Shimamura",
    "ライトオン": "Right-on",
    "青山": "Aoyama",
    "洋服の青山": "Aoyama Fashion",
    "アオキ": "AOKI",
    "スーツセレクト": "Suit Select",
    "オリヒカ": "ORIHICA",
    "モードオフ": "Mode Off",
    "セカンドストリート": "2nd Street",
    "2nd Street": "2nd Street",
    "ブックオフ": "Book Off",
    "ハードオフ": "Hard Off",
    "オフハウス": "Off House",
    "ZARA": "Zara",
    "ザラ": "Zara",
    "H&M": "H&M",
    "エイチアンドエム": "H&M",
    "GAP": "GAP",
    "ギャップ": "GAP",
    "ナイキ": "Nike",
    "アディダス": "Adidas",

    # ── Home / furniture / 100-yen ────────────────────────────────────────────
    "ニトリ": "Nitori",
    "IKEA": "IKEA",
    "イケア": "IKEA",
    "カインズ": "Cainz",
    "コーナン": "Kohnan",
    "ジョイフル本田": "Joyful Honda",
    "ドイト": "Do It",
    "コーナンホーム": "Kohnan Home",
    "ダイソー": "Daiso",
    "セリア": "Seria",
    "キャンドゥ": "Can Do",
    "ワッツ": "Watts",
    "シルク": "Silk",
    "キャン★ドゥ": "Can Do",
    "百円ショップ": "100-yen shop",
    "无印良品": "Muji",
    "無印良品": "Muji",
    "ムジルシリョウヒン": "Muji",

    # ── Online / e-commerce ───────────────────────────────────────────────────
    "アマゾン": "Amazon",
    "アマゾンジャパン": "Amazon Japan",
    "AMAZON": "Amazon",
    "AMAZON.CO.JP": "Amazon Japan",
    "アマゾンウェブサービス": "Amazon Web Services",
    "AWS": "Amazon Web Services",
    "楽天": "Rakuten",
    "楽天市場": "Rakuten Ichiba",
    "楽天ペイ": "Rakuten Pay",
    "ヤフーショッピング": "Yahoo Shopping",
    "ヤフオク": "Yahoo Auction",
    "メルカリ": "Mercari",
    "PayPay": "PayPay",
    "ペイペイ": "PayPay",

    # ── Subscriptions / digital services ─────────────────────────────────────
    "ネットフリックス": "Netflix",
    "ネトフリ": "Netflix",
    "スポティファイ": "Spotify",
    "Spotify Japan": "Spotify",
    "SPOTIFY": "Spotify",
    "ディズニープラス": "Disney+",
    "Disney+": "Disney+",
    "ディズニー": "Disney+",
    "ユーチューブプレミアム": "YouTube Premium",
    "YouTubeプレミアム": "YouTube Premium",
    "アマゾンプライム": "Amazon Prime",
    "Amazon Prime": "Amazon Prime",
    "アップルミュージック": "Apple Music",
    "Apple Music": "Apple Music",
    "アイクラウド": "iCloud",
    "iCloud": "iCloud",
    "グーグルワン": "Google One",
    "Google One": "Google One",
    "グーグルプレミアム": "Google Premium",
    "グーグルクラウドコンピュート": "Google Cloud Compute",
    "Google Cloud": "Google Cloud",
    "グーグル": "Google",
    "アドビ": "Adobe",
    "Adobe": "Adobe",
    "Adobe Creative Cloud": "Adobe Creative Cloud",
    "アドビクリエイティブクラウド": "Adobe Creative Cloud",
    "マイクロソフト": "Microsoft",
    "Microsoft": "Microsoft",
    "Microsoft 365": "Microsoft 365",
    "ドロップボックス": "Dropbox",
    "Dropbox": "Dropbox",
    "クラウドワークス": "CrowdWorks",
    "ランサーズ": "Lancers",
    "Zoom": "Zoom",
    "ズーム": "Zoom",
    "Slack": "Slack",
    "スラック": "Slack",
    "LINE Pay": "LINE Pay",
    "ラインペイ": "LINE Pay",

    # ── Telecom / mobile ──────────────────────────────────────────────────────
    "ソフトバンク": "SoftBank",
    "ドコモ": "NTT Docomo",
    "NTTドコモ": "NTT Docomo",
    "エーユー": "au",
    "AU": "au",
    "楽天モバイル": "Rakuten Mobile",
    "ワイモバイル": "Y!mobile",
    "UQモバイル": "UQ Mobile",
    "ビッグローブ": "BIGLOBE",
    "OCNモバイル": "OCN Mobile",
    "IIJmio": "IIJmio",

    # ── Public transport ──────────────────────────────────────────────────────
    "JR東日本": "JR East",
    "JR西日本": "JR West",
    "JR東海": "JR Central",
    "JR九州": "JR Kyushu",
    "JR北海道": "JR Hokkaido",
    "JR四国": "JR Shikoku",
    "JR ヒガシニホン": "JR East",
    "JR ニシニホン": "JR West",
    "東京メトロ": "Tokyo Metro",
    "都営": "Toei",
    "東急電鉄": "Tokyu Railways",
    "小田急電鉄": "Odakyu",
    "京王電鉄": "Keio Railways",
    "西武鉄道": "Seibu Railways",
    "東武鉄道": "Tobu Railways",
    "京浜急行": "Keikyu",
    "阪急電鉄": "Hankyu",
    "近畿日本鉄道": "Kintetsu",
    "名古屋市営地下鉄": "Nagoya City Subway",
    "大阪メトロ": "Osaka Metro",
    "モバイルスイカ": "Mobile Suica",
    "スイカ": "Suica",
    "パスモ": "Pasmo",
    "イコカ": "ICOCA",
    "マナカ": "manaca",
    "ニモカ": "nimoca",
    "はやかけん": "Hayakaken",

    # ── Taxi / ride-share ─────────────────────────────────────────────────────
    "タクシー": "Taxi",
    "日本交通": "Nihon Kotsu",
    "帝都自動車交通": "Teito Taxi",
    "東京無線タクシー": "Tokyo Musen Taxi",
    "Go": "GO Taxi",
    "ウーバー": "Uber",
    "Uber": "Uber",
    "S.RIDE": "S.Ride",
    "Didi": "DiDi",
    "ディディ": "DiDi",

    # ── Fuel / parking ────────────────────────────────────────────────────────
    "ENEOS": "ENEOS",
    "エネオス": "ENEOS",
    "出光興産": "Idemitsu",
    "コスモ石油": "Cosmo Oil",
    "シェル": "Shell",
    "エッソ": "Esso",
    "エネルギー": "Energy",
    "ガソリン": "Gasoline",
    "タイムズパーキング": "Times Parking",
    "タイムズ": "Times Parking",
    "NPC24H": "NPC Parking",
    "リパーク": "Repark",
    "コインパーキング": "Coin Parking",

    # ── Finance / ATM / insurance ─────────────────────────────────────────────
    "ゆうちょ銀行": "Japan Post Bank",
    "郵便局": "Japan Post",
    "みずほ銀行": "Mizuho Bank",
    "三菱UFJ銀行": "MUFG Bank",
    "三井住友銀行": "SMBC",
    "りそな銀行": "Resona Bank",
    "楽天銀行": "Rakuten Bank",
    "住信SBIネット銀行": "SBI Net Bank",
    "ジャパンネット銀行": "PayPay Bank",
    "ソニー銀行": "Sony Bank",
    "オリックス生命": "Orix Life Insurance",
    "明治安田生命": "Meiji Yasuda Life",
    "日本生命": "Nippon Life",
    "楽天カード": "Rakuten Card",
    "楽天ゴールドカード": "Rakuten Gold Card",
    "セゾンカード": "Saison Card",
    "イオンカード": "AEON Card",
    "三井住友カード": "SMBC Card",
    "JCB": "JCB",
    "VISA": "Visa",
    "Mastercard": "Mastercard",

    # ── Healthcare / wellness ─────────────────────────────────────────────────
    "病院": "Hospital",
    "クリニック": "Clinic",
    "歯科": "Dental Clinic",
    "眼科": "Eye Clinic",
    "皮膚科": "Dermatology",
    "整形外科": "Orthopedics",
    "内科": "Internal Medicine",
    "ゴールドジム": "Gold's Gym",
    "エニタイムフィットネス": "Anytime Fitness",
    "コナミスポーツ": "Konami Sports",
    "ジェクサー": "JEXER",
    "ルネサンス": "Renaissance",
    "ティップネス": "Tipness",
    "ライザップ": "RIZAP",
    "ヨガ": "Yoga Studio",

    # ── Hotels / lodging ──────────────────────────────────────────────────────
    "東横イン": "Toyoko Inn",
    "ルートイン": "Route Inn",
    "アパホテル": "APA Hotel",
    "スーパーホテル": "Super Hotel",
    "コンフォートホテル": "Comfort Hotel",
    "ホテルニューオータニ": "Hotel New Otani",
    "帝国ホテル": "Imperial Hotel",
    "Airbnb": "Airbnb",
    "エアービーアンドビー": "Airbnb",
    "じゃらん": "Jalan",
    "楽天トラベル": "Rakuten Travel",

    # ── Bookstores / education ────────────────────────────────────────────────
    "三省堂書店": "Sanseido Bookstore",
    "紀伊國屋書店": "Kinokuniya Bookstore",
    "丸善": "Maruzen",
    "文教堂": "Bunkyodo",
    "ブックファースト": "Book First",
    "コミックとらのあな": "Toranoana",
    "アニメイト": "Animate",
    "Kindle": "Kindle",
    "Udemy": "Udemy",
    "ストアカ": "Street Academy",
    "スタディサプリ": "Studysapuri",

    # ── Entertainment / leisure ───────────────────────────────────────────────
    "カラオケ": "Karaoke",
    "カラオケジョイサウンド": "Joysound Karaoke",
    "カラオケまねきねこ": "Manekineko Karaoke",
    "ビッグエコー": "Big Echo Karaoke",
    "ラウンドワン": "Round One",
    "アミューズメント": "Amusement",
    "ゲームセンター": "Arcade",
    "映画館": "Cinema",
    "TOHOシネマズ": "TOHO Cinemas",
    "イオンシネマ": "AEON Cinema",
    "ユナイテッドシネマ": "United Cinemas",
    "シネコン": "Cineplex",
    "ディズニーランド": "Tokyo Disneyland",
    "ユニバーサルスタジオジャパン": "Universal Studios Japan",
    "USJ": "Universal Studios Japan",

    # ── Utility / services ────────────────────────────────────────────────────
    "東京電力": "Tokyo Electric Power (TEPCO)",
    "東京ガス": "Tokyo Gas",
    "大阪ガス": "Osaka Gas",
    "東邦ガス": "Toho Gas",
    "東北電力": "Tohoku Electric Power",
    "関西電力": "Kansai Electric Power",
    "NTT": "NTT",
    "NHK": "NHK (Public Broadcasting)",
    "水道局": "Water Bureau",
    "クリーニング": "Dry Cleaning",
    "コインランドリー": "Coin Laundry",
    "宅急便": "Kuroneko Yamato",
    "ヤマト運輸": "Kuroneko Yamato",
    "佐川急便": "Sagawa Express",
    "日本郵便": "Japan Post",
}

# ─────────────────────────────────────────────────────────────────────────────
# Lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Apply NFKC normalization + katakana hyphen fix (same as translate.normalize_japanese_text)."""
    try:
        s = unicodedata.normalize("NFKC", text)
        # Katakana hyphen fix (copies logic from services/translation.py)
        hyphens = "-‐‑–—ｰ"
        pat = re.compile(rf"([\u30A0-\u30FF])[{hyphens}]([\u30A0-\u30FF])")
        prev = None
        while prev != s:
            prev = s
            s = pat.sub(r"\1ー\2", s)
        return re.sub(r"\s+", " ", s).strip()
    except Exception:
        return text


def lookup_merchant(raw_text: str) -> Optional[str]:
    """Return the English display name for a raw statement description, or None.

    Matching strategy (cheapest to most expensive):
    1. Exact match after NFKC normalization.
    2. Prefix match — the statement text *starts with* a library key
       (handles "スターバックスコーヒー 渋谷" matching "スターバックス").
    3. Substring match — a library key appears anywhere in the text
       (handles "LAWSON ミゾノグチエキマエ" matching "LAWSON").

    The longest matching key wins to avoid false-positive short matches.
    """
    if not raw_text or not raw_text.strip():
        return None

    normalized = _normalize(raw_text)
    normalized_upper = normalized.upper()

    # 1. Exact match
    if normalized in MERCHANT_LIBRARY:
        return MERCHANT_LIBRARY[normalized]

    # Case-insensitive exact match for ASCII-heavy entries (AMAZON, NETFLIX, etc.)
    for key, en in MERCHANT_LIBRARY.items():
        if key.upper() == normalized_upper:
            return en

    # 2 & 3. Prefix / substring — prefer the longest matching key.
    # Short purely-ASCII keys (e.g. "AWS", "au") can match inside unrelated words
    # (e.g. "LAWSON" contains "AWS").  We allow short keys only when they consist
    # of Japanese characters (which do not appear as substrings of other words).
    _ascii_only = re.compile(r'^[A-Za-z0-9 .&\-/+@_!?]+$')
    MIN_ASCII_KEY_LEN = 4

    best_key: Optional[str] = None
    best_len = 0
    for key in MERCHANT_LIBRARY:
        norm_key = _normalize(key)
        if len(norm_key) <= best_len:
            continue
        # Reject short ASCII-only keys to avoid false-positive substring matches
        if _ascii_only.match(norm_key) and len(norm_key) < MIN_ASCII_KEY_LEN:
            continue
        if normalized.startswith(norm_key) or norm_key in normalized:
            best_key = key
            best_len = len(norm_key)

    if best_key is not None:
        base_en = MERCHANT_LIBRARY[best_key]
        # Append any trailing location info (e.g. "Shibuya", "Mizonoguchi")
        suffix = normalized[best_len:].strip(" 　・-–—/")
        if suffix:
            return f"{base_en} {suffix}"
        return base_en

    return None


def apply_merchant_library(texts: list[str]) -> tuple[dict, list[str]]:
    """Resolve as many texts as possible from the library.

    Returns:
        resolved   — {raw_text: english_name} for library hits
        unresolved — list of texts not in the library (need DB cache / AI)
    """
    resolved: dict = {}
    unresolved: list = []
    for text in texts:
        hit = lookup_merchant(text)
        if hit is not None:
            resolved[text] = hit
        else:
            unresolved.append(text)
    return resolved, unresolved


def merchant_library_size() -> int:
    """Number of entries in the static library."""
    return len(MERCHANT_LIBRARY)


def iter_library_pairs():
    """Yield (jp_text, en_text) for every library entry (for DB seeding)."""
    for jp, en in MERCHANT_LIBRARY.items():
        yield jp, en
