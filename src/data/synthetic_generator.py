"""
Raw data generator — produces 4 realistic, messy tables mimicking production DB pulls.
Calibrated from:
  - FHI (Zindi): data/raw/Train.csv
  - UCI Default Credit Card: data/raw/default of credit card clients.xls
"""

import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

COUNTRIES       = ["Kenya", "Nigeria", "Tanzania", "Ghana", "Uganda", "Rwanda", "Morocco", "Senegal"]
COUNTRY_WEIGHTS = [0.22,    0.20,      0.15,       0.12,   0.10,     0.08,     0.08,      0.05]

COUNTRY_PHONE_PREFIX = {
    "Kenya": "+254", "Nigeria": "+234", "Tanzania": "+255", "Ghana": "+233",
    "Uganda": "+256", "Rwanda": "+250", "Morocco": "+212", "Senegal": "+221",
}
COUNTRY_CURRENCY = {
    "Kenya": "KES", "Nigeria": "NGN", "Tanzania": "TZS", "Ghana": "GHS",
    "Uganda": "UGX", "Rwanda": "RWF", "Morocco": "MAD", "Senegal": "XOF",
}
USD_TO_LOCAL = {
    "Kenya": 130, "Nigeria": 1500, "Tanzania": 2500, "Ghana": 12,
    "Uganda": 3700, "Rwanda": 1200, "Morocco": 10, "Senegal": 600,
}
COUNTRY_REGIONS = {
    "Kenya":    ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Nyeri", "Thika", "Meru"],
    "Nigeria":  ["Lagos", "Abuja", "Kano", "Ibadan", "Port Harcourt", "Kaduna", "Enugu", "Benin City"],
    "Tanzania": ["Dar es Salaam", "Arusha", "Dodoma", "Mwanza", "Zanzibar", "Mbeya", "Morogoro"],
    "Ghana":    ["Accra", "Kumasi", "Tamale", "Cape Coast", "Takoradi", "Sunyani", "Ho"],
    "Uganda":   ["Kampala", "Gulu", "Mbarara", "Jinja", "Entebbe", "Lira", "Mbale"],
    "Rwanda":   ["Kigali", "Huye", "Musanze", "Rubavu", "Gicumbi", "Nyagatare"],
    "Morocco":  ["Casablanca", "Rabat", "Marrakech", "Fes", "Tangier", "Agadir", "Meknes", "Oujda"],
    "Senegal":  ["Dakar", "Thiès", "Kaolack", "Ziguinchor", "Saint-Louis", "Touba"],
}
COUNTRY_URBAN_WEIGHTS = {
    "Kenya":    [0.45, 0.25, 0.30], "Nigeria": [0.55, 0.20, 0.25],
    "Tanzania": [0.30, 0.25, 0.45], "Ghana":   [0.50, 0.20, 0.30],
    "Uganda":   [0.30, 0.25, 0.45], "Rwanda":  [0.25, 0.30, 0.45],
    "Morocco":  [0.65, 0.20, 0.15], "Senegal": [0.45, 0.25, 0.30],
}

# Calibrated from FHI (Eswatini/Malawi/Lesotho ≈ baseline) + known statistics
MOBILE_MONEY_RATE = {
    "Kenya": 0.85, "Nigeria": 0.45, "Tanzania": 0.65, "Ghana": 0.55,
    "Uganda": 0.60, "Rwanda": 0.70, "Morocco": 0.38, "Senegal": 0.52,
}
BANK_ACCOUNT_RATE = {
    "Kenya": 0.42, "Nigeria": 0.38, "Tanzania": 0.25, "Ghana": 0.35,
    "Uganda": 0.22, "Rwanda": 0.28, "Morocco": 0.55, "Senegal": 0.20,
}
LOAN_ACCOUNT_RATE = {
    "Kenya": 0.18, "Nigeria": 0.12, "Tanzania": 0.08, "Ghana": 0.15,
    "Uganda": 0.10, "Rwanda": 0.14, "Morocco": 0.20, "Senegal": 0.09,
}
FEMALE_RATE = {
    "Kenya": 0.52, "Nigeria": 0.40, "Tanzania": 0.48, "Ghana": 0.45,
    "Uganda": 0.50, "Rwanda": 0.55, "Morocco": 0.35, "Senegal": 0.48,
}

SECTORS        = ["agriculture", "retail", "food_processing", "services",
                  "transport", "construction", "textile", "tech"]
SECTOR_WEIGHTS = [0.25, 0.28, 0.12, 0.14, 0.07, 0.05, 0.05, 0.04]

REJECTION_REASONS = [
    "insufficient_revenue", "no_collateral",
    "poor_credit_history", "documentation_missing", "sector_not_eligible",
]
CHANNELS = ["branch", "mobile_app", "agent", "partner"]

_NAME_FIRST = {
    "Kenya":    ["Jambo", "Karibu", "Simba", "Bora", "Kilimo", "Safari", "Amani",
                 "Harambee", "Nguvu", "Furaha", "Uzuri", "Mama", "Uhuru", "Tumaini"],
    "Nigeria":  ["Eko", "Lagos", "Obinna", "Emeka", "Tunde", "Kola", "Ngozi",
                 "Amaka", "Chidi", "Ade", "Naija", "Victory", "Grace", "Emmanuel"],
    "Tanzania": ["Karibu", "Bora", "Amani", "Kilimanjaro", "Mwanga", "Uhuru",
                 "Jua", "Pwani", "Serengeti", "Baraka", "Neema", "Selous"],
    "Ghana":    ["Akwaaba", "Sankofa", "Osei", "Mensah", "Asante", "Abena",
                 "Kwame", "Ama", "Kofi", "Adjoa", "Golden", "Adinkra"],
    "Uganda":   ["Pearl", "Kampala", "Crane", "Nile", "Bwindi", "Amara",
                 "Blessed", "Hope", "Grace", "Victory", "Kigezi", "Gorilla"],
    "Rwanda":   ["Ubumwe", "Urumuri", "Inzira", "Inyange", "Isange", "Impala",
                 "Gorilla", "Volcano", "Kigali", "Hope", "Agaseke"],
    "Morocco":  ["Atlas", "Sahara", "Maghreb", "Hassan", "Rachid", "Youssef",
                 "Fatima", "Andalous", "Alaoui", "Amina", "Zineb", "Tariq"],
    "Senegal":  ["Teranga", "Baobab", "Cheikh", "Oumar", "Fatou", "Wolof",
                 "Aminata", "Moussa", "Ibrahima", "Djoloff", "Dakar"],
}
_NAME_TYPE   = ["Traders", "Enterprises", "Agro", "General Store", "Foods",
                "Services", "Solutions", "Hardware", "Textiles", "Electronics",
                "Bakery", "Farm", "Transport", "Construction", "Tech", "Supplies"]
_NAME_SUFFIX = ["Ltd", "Limited", "& Sons", "SARL", "& Associates", "Group",
                "Co.", "& Partners", "S.A.", ""]


# ── Generator class ────────────────────────────────────────────────────────────

class RawDataGenerator:
    """
    Generates 4 raw production-like tables:
      1. sme_profiles            — business registry pull
      2. sme_financial_profile   — credit bureau pull
      3. product_interactions    — CRM event log (NOT a matrix)
      4. product_catalog         — static product reference
    """

    def __init__(self, data_raw_path: str = "data/raw", random_state: int = 42):
        self.raw_path  = Path(data_raw_path)
        self.out_path  = Path("data/raw")
        self.rng       = np.random.default_rng(random_state)
        self.cal       = {}          # calibration stats
        self.sme_df:   pd.DataFrame = None
        self.fin_df:   pd.DataFrame = None
        self.inter_df: pd.DataFrame = None
        self.catalog:  pd.DataFrame = None

    # ── Calibration ────────────────────────────────────────────────────────────

    def calibrate_from_datasets(self) -> "RawDataGenerator":
        logger.info("Reading real datasets for calibration...")
        self._calibrate_fhi()
        self._calibrate_uci()
        self._print_calibration_report()
        return self

    def _calibrate_fhi(self):
        path = self.raw_path / "Train.csv"
        if not path.exists():
            logger.warning("FHI Train.csv not found — using hardcoded defaults.")
            self.cal["fhi_loaded"] = False
            self._fhi_defaults()
            return

        fhi = pd.read_csv(path)
        logger.info(f"FHI loaded: {len(fhi):,} rows | countries: {sorted(fhi['country'].unique())}")

        # Business age (FHI: mean=7.0, std=7.6, missing=2.6%)
        age = fhi["business_age_years"].dropna()
        self.cal.update({
            "biz_age_mean":    float(age.mean()),
            "biz_age_std":     float(age.std()),
            "biz_age_missing": float(fhi["business_age_years"].isna().mean()),
        })

        # Revenue — local currency, log-normal
        rev = fhi["business_turnover"][fhi["business_turnover"] > 100].dropna()
        self.cal.update({
            "rev_log_mean":  float(np.log(rev).mean()),
            "rev_log_std":   float(np.log(rev).std()),
            "rev_missing":   float(fhi["business_turnover"].isna().mean()),
            "rev_p25":       float(np.percentile(rev, 25)),
            "rev_p50":       float(np.percentile(rev, 50)),
            "rev_p75":       float(np.percentile(rev, 75)),
        })

        # Expense ratio (FHI: mean=0.59, std=0.47)
        mask = (fhi["business_turnover"] > 0) & (fhi["business_expenses"] > 0)
        ratio = (fhi.loc[mask, "business_expenses"] / fhi.loc[mask, "business_turnover"]).clip(0, 2)
        self.cal.update({
            "exp_ratio_mean": float(ratio.mean()),
            "exp_ratio_std":  float(ratio.std()),
            "exp_missing":    float(fhi["business_expenses"].isna().mean()),
        })

        # Owner age (FHI: mean=41.7, std=13.3)
        oa = fhi["owner_age"].dropna()
        self.cal.update({
            "owner_age_mean":    float(oa.mean()),
            "owner_age_std":     float(oa.std()),
            "owner_age_missing": float(fhi["owner_age"].isna().mean()),
        })

        # Financial behaviour rates (FHI string-valued)
        self.cal["keeps_records_yes"]    = float((fhi["keeps_financial_records"] == "Yes, always").mean())
        self.cal["keeps_records_partial"] = float(fhi["keeps_financial_records"].isin(
            ["Yes, sometimes", "Yes"]).mean())
        self.cal["tax_compliant"]         = float((fhi["compliance_income_tax"] == "Yes").mean())
        self.cal["uses_informal_lender"]  = float(fhi["uses_informal_lender"].isin(
            ["Have now", "Used to have but don't have now"]).mean())
        self.cal["uses_friends_family"]   = float(fhi["uses_friends_family_savings"].isin(
            ["Have now", "Used to have but don't have now"]).mean())
        self.cal["has_cash_flow_problem"] = float((fhi["current_problem_cash_flow"] == "Yes").mean())
        self.cal["has_insurance_rate"]    = float((fhi["has_insurance"] == "Yes").mean())
        self.cal["has_internet_banking"]  = float((fhi["has_internet_banking"] == "Have now").mean())
        self.cal["has_credit_card"]       = float((fhi["has_credit_card"] == "Have now").mean())

        # Target distribution (Low/Medium/High financial health)
        tgt = fhi["Target"].value_counts(normalize=True)
        self.cal["target_low"]    = float(tgt.get("Low", 0.653))
        self.cal["target_medium"] = float(tgt.get("Medium", 0.298))
        self.cal["target_high"]   = float(tgt.get("High", 0.049))
        self.cal["fhi_loaded"] = True

    def _fhi_defaults(self):
        self.cal.update({
            "biz_age_mean": 7.0, "biz_age_std": 7.6, "biz_age_missing": 0.026,
            "rev_log_mean": 9.55, "rev_log_std": 2.94, "rev_missing": 0.022,
            "rev_p25": 1800, "rev_p50": 6000, "rev_p75": 60000,
            "exp_ratio_mean": 0.59, "exp_ratio_std": 0.47, "exp_missing": 0.024,
            "owner_age_mean": 41.7, "owner_age_std": 13.3, "owner_age_missing": 0.02,
            "keeps_records_yes": 0.149, "keeps_records_partial": 0.218,
            "tax_compliant": 0.130, "uses_informal_lender": 0.399,
            "uses_friends_family": 0.331, "has_cash_flow_problem": 0.398,
            "has_insurance_rate": 0.037, "has_internet_banking": 0.098,
            "has_credit_card": 0.037,
            "target_low": 0.653, "target_medium": 0.298, "target_high": 0.049,
        })

    def _calibrate_uci(self):
        path = self.raw_path / "default of credit card clients.xls"
        if not path.exists():
            logger.warning("UCI .xls not found — using default_rate=0.221")
            self.cal.update({"uci_default_rate": 0.221, "uci_loaded": False})
            self.cal["pay_delay_dist"] = {-2: 0.092, -1: 0.190, 0: 0.491,
                                           1: 0.123,  2: 0.089, 3: 0.011}
            self.cal["credit_limit_log_mean"] = float(np.log(167484 / 30))
            self.cal["credit_limit_log_std"]  = 0.9
            return

        uci = pd.read_excel(path, header=1)
        logger.info(f"UCI loaded: {len(uci):,} rows")

        col = "default payment next month"
        self.cal["uci_default_rate"] = float(uci[col].mean())
        self.cal["uci_loaded"] = True

        # Payment delay distribution (PAY_0)
        pay_counts = uci["PAY_0"].value_counts(normalize=True)
        self.cal["pay_delay_dist"] = {int(k): float(v) for k, v in pay_counts.items()}

        # Credit limit in USD (TWD ÷ 30)
        limit_usd = uci["LIMIT_BAL"] / 30
        self.cal["credit_limit_log_mean"] = float(np.log(limit_usd).mean())
        self.cal["credit_limit_log_std"]  = float(np.log(limit_usd).std())

    def _print_calibration_report(self):
        c = self.cal
        logger.info("═" * 60)
        logger.info("CALIBRATION REPORT")
        logger.info(f"  Source FHI loaded:      {c.get('fhi_loaded', False)}")
        logger.info(f"  Source UCI loaded:      {c.get('uci_loaded', False)}")
        logger.info(f"  ── From FHI ─────────────────────────────────────")
        logger.info(f"  Biz age:         mean={c['biz_age_mean']:.1f}  std={c['biz_age_std']:.1f}  missing={c['biz_age_missing']:.2%}")
        logger.info(f"  Revenue log:     mean={c['rev_log_mean']:.2f}  std={c['rev_log_std']:.2f}  missing={c['rev_missing']:.2%}")
        logger.info(f"  Revenue USD p50: {c['rev_p50']:,.0f} local → scaled to USD")
        logger.info(f"  Expense ratio:   mean={c['exp_ratio_mean']:.2f}  std={c['exp_ratio_std']:.2f}")
        logger.info(f"  Owner age:       mean={c['owner_age_mean']:.1f}  std={c['owner_age_std']:.1f}")
        logger.info(f"  Tax compliant:   {c['tax_compliant']:.3f}")
        logger.info(f"  Informal lender: {c['uses_informal_lender']:.3f}")
        logger.info(f"  Cash flow prob:  {c['has_cash_flow_problem']:.3f}")
        logger.info(f"  Target Low/Med/Hi: {c['target_low']:.3f}/{c['target_medium']:.3f}/{c['target_high']:.3f}")
        logger.info(f"  ── From UCI ─────────────────────────────────────")
        logger.info(f"  Default rate:    {c['uci_default_rate']:.3f}")
        logger.info(f"  Credit lim log:  mean={c['credit_limit_log_mean']:.2f}  std={c['credit_limit_log_std']:.2f}")
        logger.info("═" * 60)

    # ── Table 1: sme_profiles ─────────────────────────────────────────────────

    def generate_sme_profiles(self, n: int = 5000) -> pd.DataFrame:
        logger.info(f"Generating {n:,} SME profiles (raw business registry)...")
        rng, c = self.rng, self.cal

        countries = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
        sectors   = rng.choice(SECTORS,   size=n, p=SECTOR_WEIGHTS)

        regions, urban_rural = [], []
        for ctry in countries:
            regions.append(rng.choice(COUNTRY_REGIONS[ctry]))
            urban_rural.append(rng.choice(
                ["urban", "peri-urban", "rural"], p=COUNTRY_URBAN_WEIGHTS[ctry]
            ))

        reg_dates = pd.to_datetime("2005-01-01") + pd.to_timedelta(
            rng.integers(0, 365 * 18, size=n), unit="D"
        )

        # Owner age — calibrated from FHI (mean=41.7, std=13.3)
        raw_age    = rng.normal(c["owner_age_mean"], c["owner_age_std"], n).clip(18, 80)
        miss_age   = rng.random(n) < c["owner_age_missing"]
        owner_age  = np.where(miss_age, np.nan, raw_age.round())

        owner_sex  = np.array([
            rng.choice(["Female", "Male"], p=[FEMALE_RATE[ct], 1 - FEMALE_RATE[ct]])
            for ct in countries
        ])
        owner_edu  = rng.choice(
            ["tertiary", "secondary", "primary", "none"], size=n, p=[0.35, 0.46, 0.15, 0.04]
        )
        biz_names  = [self._make_biz_name(ct) for ct in countries]

        # Business age — exponential skew calibrated from FHI
        raw_years  = np.abs(rng.exponential(4.5, n) + rng.uniform(0, 2, n)).clip(0, 40)
        miss_years = rng.random(n) < c["biz_age_missing"]
        years_biz  = np.where(miss_years, np.nan, raw_years.round(1))

        n_emp = np.round(np.exp(rng.normal(1.2, 0.9, n))).clip(1, 200).astype(int)

        # Revenue in USD — calibrated to African SME range $50-$200k
        rev_usd  = np.exp(rng.normal(7.3, 1.8, n)).clip(50, 500_000)
        miss_rev = rng.random(n) < 0.08
        annual_rev_usd = np.where(miss_rev, np.nan, rev_usd.round(2))

        # 20% of rows report revenue in local currency
        is_local    = rng.random(n) < 0.20
        currency    = np.where(is_local, [COUNTRY_CURRENCY[ct] for ct in countries], "USD")
        rev_reported = annual_rev_usd.copy().astype(object)
        for i in np.where(is_local)[0]:
            v = annual_rev_usd[i]
            if not (isinstance(v, float) and np.isnan(v)):
                rev_reported[i] = round(float(v) * USD_TO_LOCAL[countries[i]], -2)

        # Business expenses (calibrated from FHI: ratio mean=0.59)
        exp_ratio = rng.normal(c["exp_ratio_mean"], c["exp_ratio_std"], n).clip(0.05, 2.0)
        expenses  = rev_usd * exp_ratio
        miss_exp  = rng.random(n) < 0.10
        biz_exp_usd = np.where(miss_exp | miss_rev, np.nan, expenses.round(2))

        # Personal income (separate, calibrated from FHI)
        pers_inc    = np.exp(rng.normal(6.5, 1.5, n)).clip(20, 50_000)
        miss_inc    = rng.random(n) < 0.15
        pers_inc_usd = np.where(miss_inc, np.nan, pers_inc.round(2))

        # Financial behaviour — calibrated from FHI
        p_yes  = c["keeps_records_yes"]
        p_part = c["keeps_records_partial"]
        p_no   = max(0.01, 1 - p_yes - p_part)
        keeps_records = rng.choice(["Yes", "No", "Partially"], size=n, p=[p_yes, p_no, p_part])

        p_tax = c["tax_compliant"]
        tax_registered = rng.choice(
            ["Yes", "No", "Don't know"], size=n, p=[p_tax, 0.833, max(0, 1 - p_tax - 0.833)]
        )
        has_premises = rng.choice(["Yes", "No"], size=n, p=[0.72, 0.28])
        phones       = [self._make_phone(ct) for ct in countries]

        # Record quality → extra missing values
        record_quality = rng.choice(
            ["complete", "partial", "minimal"], size=n, p=[0.55, 0.33, 0.12]
        )
        for i in np.where(record_quality == "minimal")[0]:
            if rng.random() < 0.70: annual_rev_usd[i] = np.nan
            if rng.random() < 0.70: rev_reported[i]   = np.nan
            if rng.random() < 0.80: biz_exp_usd[i]    = np.nan
            if rng.random() < 0.50: pers_inc_usd[i]   = np.nan
            if rng.random() < 0.40: years_biz[i]      = np.nan

        sme_ids = [f"SME_{i+1:05d}" for i in range(n)]

        df = pd.DataFrame({
            "sme_id":                  sme_ids,
            "registration_date":       reg_dates,
            "country":                 countries,
            "region":                  regions,
            "urban_rural":             urban_rural,
            "sector":                  sectors,
            "owner_age":               owner_age,
            "owner_sex":               owner_sex,
            "owner_education":         owner_edu,
            "business_name":           biz_names,
            "years_in_business":       years_biz,
            "n_employees":             n_emp,
            "annual_revenue_usd":      annual_rev_usd,
            "annual_revenue_reported": rev_reported,
            "currency_reported":       currency,
            "business_expenses_usd":   biz_exp_usd,
            "personal_income_usd":     pers_inc_usd,
            "keeps_financial_records": keeps_records,
            "registered_with_tax":     tax_registered,
            "has_physical_premises":   has_premises,
            "phone_number":            phones,
            "data_source":             "business_registry",
            "record_quality":          record_quality,
        })

        # ── Inject messiness ──────────────────────────────────────────────
        # 3% duplicates (re-registration)
        n_dup   = int(n * 0.03)
        dup_idx = rng.choice(n, size=n_dup, replace=False)
        dups    = df.iloc[dup_idx].copy()
        dups["sme_id"] = [f"SME_{n+i+1:05d}" for i in range(n_dup)]
        dups["registration_date"] = dups["registration_date"] + pd.to_timedelta(
            rng.integers(30, 365, size=n_dup), unit="D"
        )
        df = pd.concat([df, dups], ignore_index=True)

        # 5% inconsistent: expenses > revenue
        incon = rng.choice(df.index, size=int(len(df) * 0.05), replace=False)
        for idx in incon:
            rev = df.at[idx, "annual_revenue_usd"]
            if not (isinstance(rev, float) and np.isnan(rev)):
                df.at[idx, "business_expenses_usd"] = float(rev) * rng.uniform(1.05, 2.5)

        df = df.reset_index(drop=True)
        self.sme_df = df
        logger.success(
            f"sme_profiles: {len(df):,} rows | "
            f"rev missing={df['annual_revenue_usd'].isna().mean():.1%} | "
            f"duplicates={n_dup}"
        )
        return df

    def _make_biz_name(self, country: str) -> str:
        first  = self.rng.choice(_NAME_FIRST[country])
        btype  = self.rng.choice(_NAME_TYPE)
        suffix = self.rng.choice(_NAME_SUFFIX)
        return f"{first} {btype} {suffix}".strip()

    def _make_phone(self, country: str) -> str:
        prefix = COUNTRY_PHONE_PREFIX[country]
        digits = "".join(str(d) for d in self.rng.integers(0, 10, size=9))
        return f"{prefix}{digits}"

    # ── Table 2: sme_financial_profile ───────────────────────────────────────

    def generate_financial_profiles(self) -> pd.DataFrame:
        if self.sme_df is None:
            raise RuntimeError("Call generate_sme_profiles() first.")
        c   = self.cal
        rng = self.rng

        # 80% coverage (15% of SMEs have no bureau record)
        all_ids   = self.sme_df["sme_id"].tolist()
        n_bureau  = int(len(all_ids) * 0.85)
        bureau_ids_sme = rng.choice(all_ids, size=n_bureau, replace=False)
        n = len(bureau_ids_sme)
        logger.info(f"Generating financial profiles: {n:,} rows (85% SME coverage)...")

        sme_lookup = self.sme_df.set_index("sme_id")

        pull_dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(
            rng.integers(0, 365 * 4, size=n), unit="D"
        )

        # Financial inclusion — per country
        def _pick_status(rate, sometimes_rate=0.05):
            p_have = rate
            p_used = sometimes_rate
            p_never = max(0, 1 - p_have - p_used)
            return rng.choice(
                ["Have now", "Never had", "Used to have but don't have now"],
                p=[p_have, p_never, p_used]
            )

        has_bank, has_mobile, has_loan = [], [], []
        for sid in bureau_ids_sme:
            ctry = sme_lookup.loc[sid, "country"] if sid in sme_lookup.index else "Kenya"
            has_bank.append(_pick_status(BANK_ACCOUNT_RATE.get(ctry, 0.30)))
            has_mobile.append(_pick_status(MOBILE_MONEY_RATE.get(ctry, 0.60)))
            has_loan.append(_pick_status(LOAN_ACCOUNT_RATE.get(ctry, 0.12)))

        has_credit_card = [
            _pick_status(c.get("has_credit_card", 0.037), 0.010) for _ in range(n)
        ]
        has_debit_card = [
            "Have now" if hb == "Have now"
            else rng.choice(["Never had", "Have now"], p=[0.85, 0.15])
            for hb in has_bank
        ]
        has_internet = [
            _pick_status(c.get("has_internet_banking", 0.098), 0.054) for _ in range(n)
        ]

        # Previous loans and defaults — calibrated from UCI (22.1%)
        default_rate  = c.get("uci_default_rate", 0.221)
        n_prev_loans  = rng.poisson(lam=1.5, size=n)
        n_defaults    = np.array([
            rng.binomial(max(0, nl), default_rate) for nl in n_prev_loans
        ])
        # Inject 2% impossible: n_defaults > n_prev_loans
        bad_idx = rng.choice(n, size=int(n * 0.02), replace=False)
        for i in bad_idx:
            if n_prev_loans[i] > 0:
                n_defaults[i] = n_prev_loans[i] + 1

        # Borrowed amounts — calibrated from UCI LIMIT_BAL ÷ 30 → USD
        lm = c["credit_limit_log_mean"]
        ls = c["credit_limit_log_std"]
        total_borrowed = np.where(
            n_prev_loans > 0,
            np.exp(rng.normal(lm, ls, n)).clip(50, 200_000),
            0.0
        ).round(2)
        outstanding = np.where(
            n_prev_loans > 0,
            total_borrowed * rng.uniform(0, 0.8, n),
            0.0
        ).round(2)

        # Worst payment delay — calibrated from UCI PAY_0 distribution
        pay_dist   = c.get("pay_delay_dist", {-2: 0.092, -1: 0.190, 0: 0.491, 1: 0.123, 2: 0.089})
        delay_vals = list(pay_dist.keys())
        delay_prob = np.array(list(pay_dist.values()), dtype=float)
        delay_prob /= delay_prob.sum()
        pay_codes  = rng.choice(delay_vals, size=n, p=delay_prob)
        delay_map  = {-2: 0, -1: 0, 0: 0, 1: 30, 2: 60, 3: 90, 4: 120, 5: 150, 6: 180, 7: 180, 8: 180}
        worst_delay = np.where(
            n_prev_loans == 0, np.nan,
            [float(delay_map.get(int(p), 0)) for p in pay_codes]
        )

        # Informal finance — calibrated from FHI
        p_il = c.get("uses_informal_lender", 0.399)
        uses_informal = rng.choice(
            ["Yes", "No", "Sometimes"], size=n,
            p=[p_il * 0.12, 1 - p_il, p_il * 0.88]
        )
        p_ff = c.get("uses_friends_family", 0.331)
        uses_ff = rng.choice(
            ["Yes", "No", "Sometimes"], size=n,
            p=[p_ff * 0.35, 1 - p_ff, p_ff * 0.65]
        )

        # Collateral
        collateral_type = rng.choice(
            ["none", "land", "equipment", "inventory", "guarantor"],
            size=n, p=[0.55, 0.15, 0.12, 0.10, 0.08]
        )
        collateral_val = np.where(
            collateral_type != "none",
            np.exp(rng.normal(8.0, 1.5, n)).clip(100, 500_000).round(2),
            np.nan
        )

        # Bureau score: NULL for unbanked (structural missing, not random)
        bureau_score = np.where(
            np.array(has_bank) == "Never had",
            np.nan,
            np.clip(rng.normal(580, 120, n), 300, 850).round()
        )
        # Extra 10% NaN (partial data)
        bureau_score = np.where(rng.random(n) < 0.10, np.nan, bureau_score)

        bureau_ids = [
            "BUR_" + "".join(rng.choice(list(string.ascii_uppercase + string.digits), size=8))
            for _ in range(n)
        ]

        df = pd.DataFrame({
            "bureau_id":                   bureau_ids,
            "sme_id":                      bureau_ids_sme,
            "pull_date":                   pull_dates,
            "has_bank_account":            has_bank,
            "has_mobile_money":            has_mobile,
            "has_loan_account":            has_loan,
            "has_credit_card":             has_credit_card,
            "has_debit_card":              has_debit_card,
            "has_internet_banking":        has_internet,
            "n_previous_loans":            n_prev_loans,
            "n_defaults":                  n_defaults,
            "total_borrowed_usd":          total_borrowed,
            "current_outstanding_usd":     outstanding,
            "worst_payment_delay_days":    worst_delay,
            "uses_informal_lender":        uses_informal,
            "uses_friends_family_savings": uses_ff,
            "collateral_type":             collateral_type,
            "collateral_value_usd":        collateral_val,
            "bureau_score":                bureau_score,
        })

        self.fin_df = df
        logger.success(
            f"sme_financial_profile: {len(df):,} rows | "
            f"bureau_score null={pd.isna(bureau_score).mean():.1%} | "
            f"default_rate={n_defaults.clip(0, n_prev_loans).sum()/max(n_prev_loans.sum(),1):.3f}"
        )
        return df

    # ── Table 3: product_interactions ────────────────────────────────────────

    def generate_product_interactions(self) -> pd.DataFrame:
        if self.sme_df is None or self.fin_df is None:
            raise RuntimeError("Call generate_sme/fin_profiles() first.")
        logger.info("Generating product interactions (raw CRM log)...")
        rng = self.rng

        sme_lkp = self.sme_df.set_index("sme_id")
        fin_lkp = self.fin_df.set_index("sme_id")
        rows = []
        iid  = 1

        for sme_id in self.sme_df["sme_id"]:
            if sme_id not in sme_lkp.index:
                continue
            sme      = sme_lkp.loc[sme_id]
            fin      = fin_lkp.loc[sme_id] if sme_id in fin_lkp.index else None
            reg_date = pd.to_datetime(sme["registration_date"])
            probs    = self._product_probs(sme, fin)

            for prod_id in range(1, 9):
                roll = rng.random()
                p    = probs[prod_id - 1]

                if roll < p:
                    chain = self._adoption_chain(sme_id, prod_id, sme, fin, reg_date, rng)
                    rows.extend(chain)
                    iid += len(chain)
                elif roll < p + 0.07:
                    rows.append(self._rejected_row(sme_id, prod_id, sme, fin, reg_date, rng))
                    iid += 1
                elif roll < p + 0.10:
                    rows.append(self._inquiry_row(sme_id, prod_id, reg_date, rng))
                    iid += 1

        df = pd.DataFrame(rows)

        # ── Messiness injection ───────────────────────────────────────────
        # 8% NULL satisfaction where it should exist
        has_score = df["satisfaction_score"].notna()
        null_idx  = df[has_score].sample(frac=0.08, random_state=1).index
        df.loc[null_idx, "satisfaction_score"] = np.nan

        # 3% duplicate rows
        n_dup = int(len(df) * 0.03)
        dups  = df.sample(n=n_dup, random_state=2).copy()
        df    = pd.concat([df, dups], ignore_index=True)

        # 1% interaction date before registration
        n_early = int(len(df) * 0.01)
        for idx in rng.choice(df.index, size=n_early, replace=False):
            sid = df.at[idx, "sme_id"]
            if sid in sme_lkp.index:
                reg = pd.to_datetime(sme_lkp.loc[sid, "registration_date"])
                df.at[idx, "interaction_date"] = reg - pd.to_timedelta(
                    int(rng.integers(10, 180)), unit="D"
                )

        # 2% approved > requested (data entry error)
        appr_mask = df["amount_approved_usd"].notna() & df["amount_requested_usd"].notna()
        for idx in df[appr_mask].sample(frac=0.02, random_state=3).index:
            df.at[idx, "amount_approved_usd"] = df.at[idx, "amount_requested_usd"] * rng.uniform(1.05, 1.30)

        df = df.reset_index(drop=True)
        df.insert(0, "interaction_id", range(1, len(df) + 1))

        self.inter_df = df
        n_pairs   = df.groupby(["sme_id", "product_id"]).ngroups
        n_smes_u  = self.sme_df["sme_id"].nunique()
        sparsity  = 1 - n_pairs / (n_smes_u * 8)
        logger.success(
            f"product_interactions: {len(df):,} rows | "
            f"unique SME-product pairs={n_pairs:,} | sparsity≈{sparsity:.1%}"
        )
        return df

    def _product_probs(self, sme, fin) -> list:
        """8 adoption probabilities [P1..P8] based on SME profile."""
        sector = sme["sector"]
        rev    = float(sme["annual_revenue_usd"]) if not _isnan(sme["annual_revenue_usd"]) else 800.0
        has_mobile = fin is not None and fin["has_mobile_money"] == "Have now"
        has_default = fin is not None and not _isnan(fin["n_defaults"]) and float(fin["n_defaults"]) > 0

        # Base [microcredit3m, microcredit12m, agri_ins, equip_lease,
        #        group_sav, mobile_pay, invoice_fin, crop_adv]
        p = [0.12, 0.06, 0.04, 0.03, 0.10, 0.12, 0.02, 0.04]

        if sector == "agriculture":
            p[2] += 0.35; p[7] += 0.45; p[0] += 0.18; p[4] += 0.25
        elif sector == "retail":
            p[0] += 0.38; p[1] += 0.10
            p[5] += 0.30 if has_mobile else 0.10
            if rev > 5000: p[6] += 0.08
        elif sector == "food_processing":
            p[0] += 0.25; p[3] += 0.08
            if rev > 5000: p[6] += 0.12
        elif sector == "services":
            p[0] += 0.25; p[1] += 0.08
            p[5] += 0.25 if has_mobile else 0.05
        elif sector == "transport":
            p[3] += 0.22; p[1] += 0.12; p[0] += 0.12
        elif sector == "construction":
            p[3] += 0.18; p[1] += 0.15
            if rev > 5000: p[6] += 0.10
        elif sector == "textile":
            p[3] += 0.12; p[0] += 0.20; p[4] += 0.10
        elif sector == "tech":
            p[5] += 0.35 if has_mobile else 0.10; p[1] += 0.12
            if rev > 5000: p[6] += 0.15

        if rev > 10_000:
            p[3] += 0.25; p[6] += 0.20; p[1] += 0.15
        if rev > 30_000:
            p[6] += 0.10
        if has_mobile:
            p[5] += 0.35
        if has_default:
            p[0] = max(0.02, p[0] - 0.10)
            p[1] = max(0.01, p[1] - 0.08)
            p[7] = max(0.02, p[7] - 0.05)

        return [min(0.90, max(0.01, x)) for x in p]

    def _adoption_chain(self, sme_id, prod_id, sme, fin, reg_date, rng) -> list:
        credit = prod_id in {1, 2, 7, 8}
        has_mobile = fin is not None and fin["has_mobile_money"] == "Have now"
        has_default = fin is not None and not _isnan(fin["n_defaults"]) and float(fin["n_defaults"]) > 0

        channel = "mobile_app" if (has_mobile and rng.random() < 0.55) else \
                  rng.choice(CHANNELS, p=[0.45, 0.20, 0.25, 0.10])

        days_offset = int(rng.integers(30, 365 * 5))
        app_date    = reg_date + pd.to_timedelta(days_offset, unit="D")

        amt_req = float(np.exp(rng.normal(7.5, 1.2)).clip(100, 50_000)) if credit else np.nan
        amt_app = float(amt_req * rng.uniform(0.70, 1.0)) if credit else np.nan

        rows = [
            _irow(sme_id, prod_id, "application", app_date,
                  amt_req, np.nan, np.nan, np.nan, np.nan, channel),
            _irow(sme_id, prod_id, "approved",
                  app_date + pd.to_timedelta(int(rng.integers(1, 14)), unit="D"),
                  amt_req, amt_app, np.nan, np.nan, np.nan, channel),
        ]

        if rng.random() < 0.75:
            dur = int(rng.integers(60, 400) if credit else rng.integers(30, 180))
            end = app_date + pd.to_timedelta(dur, unit="D")
            will_default = has_default and rng.random() < 0.35
            itype = "defaulted" if will_default else "completed"
            sat   = int(rng.choice([1, 2], p=[0.7, 0.3])) if will_default \
                    else int(rng.choice([3, 4, 5], p=[0.20, 0.45, 0.35]))
            rep   = float(rng.uniform(10, 60)) if (will_default and credit) \
                    else (float(rng.uniform(85, 100)) if credit else np.nan)
            rows.append(_irow(sme_id, prod_id, itype, end,
                              amt_req, amt_app, sat, rep, np.nan, channel))
        return rows

    def _rejected_row(self, sme_id, prod_id, sme, fin, reg_date, rng) -> dict:
        credit     = prod_id in {1, 2, 7, 8}
        has_mobile = fin is not None and fin["has_mobile_money"] == "Have now"
        ch = "mobile_app" if (has_mobile and rng.random() < 0.4) else \
             rng.choice(CHANNELS, p=[0.50, 0.15, 0.25, 0.10])
        days = int(rng.integers(30, 365 * 3))
        return _irow(
            sme_id, prod_id, "rejected",
            reg_date + pd.to_timedelta(days, unit="D"),
            float(np.exp(rng.normal(7.0, 1.2)).clip(100, 30_000)) if credit else np.nan,
            np.nan, np.nan, np.nan,
            rng.choice(REJECTION_REASONS), ch,
        )

    def _inquiry_row(self, sme_id, prod_id, reg_date, rng) -> dict:
        days = int(rng.integers(7, 365 * 2))
        return _irow(
            sme_id, prod_id, "inquiry",
            reg_date + pd.to_timedelta(days, unit="D"),
            np.nan, np.nan, np.nan, np.nan, np.nan,
            rng.choice(CHANNELS),
        )

    # ── Table 4: product_catalog ──────────────────────────────────────────────

    def generate_product_catalog(self) -> pd.DataFrame:
        df = pd.DataFrame([
            {"product_id": 1, "product_code": "MKT3M",    "product_name": "Microcredit 3 months",
             "category": "credit",    "risk_level": "low",    "min_revenue_usd": 500,
             "max_amount_usd": 5_000, "typical_duration_days": 90,  "interest_rate_annual_pct": 18.0,
             "requires_collateral": False, "requires_bank_account": False,
             "available_countries": "all",
             "launch_date": "2015-03-01", "is_active": True},
            {"product_id": 2, "product_code": "MKT12M",   "product_name": "Microcredit 12 months",
             "category": "credit",    "risk_level": "medium", "min_revenue_usd": 1_000,
             "max_amount_usd": 20_000, "typical_duration_days": 365, "interest_rate_annual_pct": 24.0,
             "requires_collateral": True,  "requires_bank_account": True,
             "available_countries": "all",
             "launch_date": "2015-03-01", "is_active": True},
            {"product_id": 3, "product_code": "AGRI-INS", "product_name": "Agricultural insurance",
             "category": "insurance", "risk_level": "low",    "min_revenue_usd": 200,
             "max_amount_usd": 10_000, "typical_duration_days": 180, "interest_rate_annual_pct": 0.0,
             "requires_collateral": False, "requires_bank_account": False,
             "available_countries": "Kenya,Tanzania,Uganda,Rwanda,Ghana,Nigeria,Senegal",
             "launch_date": "2017-06-01", "is_active": True},
            {"product_id": 4, "product_code": "EQ-LEASE", "product_name": "Equipment leasing",
             "category": "leasing",   "risk_level": "medium", "min_revenue_usd": 2_000,
             "max_amount_usd": 100_000, "typical_duration_days": 730, "interest_rate_annual_pct": 15.0,
             "requires_collateral": True,  "requires_bank_account": True,
             "available_countries": "all",
             "launch_date": "2016-01-01", "is_active": True},
            {"product_id": 5, "product_code": "GRP-SAV",  "product_name": "Group savings",
             "category": "savings",   "risk_level": "low",    "min_revenue_usd": 100,
             "max_amount_usd": 50_000, "typical_duration_days": 365, "interest_rate_annual_pct": 0.0,
             "requires_collateral": False, "requires_bank_account": False,
             "available_countries": "all",
             "launch_date": "2014-01-01", "is_active": True},
            {"product_id": 6, "product_code": "MOB-PAY",  "product_name": "Mobile payment setup",
             "category": "payments",  "risk_level": "low",    "min_revenue_usd": 50,
             "max_amount_usd": 500,   "typical_duration_days": 30,  "interest_rate_annual_pct": 0.0,
             "requires_collateral": False, "requires_bank_account": False,
             "available_countries": "all",
             "launch_date": "2018-01-01", "is_active": True},
            {"product_id": 7, "product_code": "INV-FIN",  "product_name": "Invoice financing",
             "category": "credit",    "risk_level": "high",   "min_revenue_usd": 5_000,
             "max_amount_usd": 150_000, "typical_duration_days": 60, "interest_rate_annual_pct": 30.0,
             "requires_collateral": False, "requires_bank_account": True,
             "available_countries": "Kenya,Nigeria,Ghana,Morocco",
             "launch_date": "2019-06-01", "is_active": True},
            {"product_id": 8, "product_code": "CROP-ADV", "product_name": "Crop advance loan",
             "category": "credit",    "risk_level": "medium", "min_revenue_usd": 300,
             "max_amount_usd": 15_000, "typical_duration_days": 120, "interest_rate_annual_pct": 20.0,
             "requires_collateral": False, "requires_bank_account": False,
             "available_countries": "Kenya,Tanzania,Uganda,Rwanda,Ghana,Nigeria,Senegal",
             "launch_date": "2016-09-01", "is_active": True},
        ])
        self.catalog = df
        return df

    # ── Orchestration ─────────────────────────────────────────────────────────

    def generate_all(self, n_smes: int = 5000) -> dict:
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.calibrate_from_datasets()

        sme   = self.generate_sme_profiles(n=n_smes)
        fin   = self.generate_financial_profiles()
        inter = self.generate_product_interactions()
        cat   = self.generate_product_catalog()

        sme.to_csv(self.out_path / "sme_profiles.csv", index=False)
        fin.to_csv(self.out_path / "sme_financial_profile.csv", index=False)
        inter.to_csv(self.out_path / "product_interactions.csv", index=False)
        cat.to_csv(self.out_path / "product_catalog.csv", index=False)

        logger.success(f"All 4 tables saved → {self.out_path}/")
        self._print_summary(sme, fin, inter, cat)
        return {"sme_profiles": sme, "sme_financial_profile": fin,
                "product_interactions": inter, "product_catalog": cat}

    def _print_summary(self, sme, fin, inter, cat):
        logger.info("═" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info(f"  sme_profiles:          {len(sme):>7,} rows × {sme.shape[1]} cols")
        logger.info(f"  sme_financial_profile: {len(fin):>7,} rows × {fin.shape[1]} cols")
        logger.info(f"  product_interactions:  {len(inter):>7,} rows × {inter.shape[1]} cols")
        logger.info(f"  product_catalog:       {len(cat):>7,} rows")
        logger.info("")
        logger.info("  Missing rates (sme_profiles):")
        for col in ["owner_age", "years_in_business", "annual_revenue_usd",
                    "business_expenses_usd", "personal_income_usd"]:
            if col in sme.columns:
                logger.info(f"    {col:<32}: {sme[col].isna().mean():.2%}")
        logger.info("")
        logger.info("  Interaction types:")
        for itype, cnt in inter["interaction_type"].value_counts().items():
            logger.info(f"    {itype:<15}: {cnt:,}")
        n_pairs  = inter.groupby(["sme_id", "product_id"]).ngroups
        n_smes_u = sme["sme_id"].nunique()
        logger.info(f"  Unique SME-product pairs: {n_pairs:,}")
        logger.info(f"  Approx matrix sparsity:   {1 - n_pairs/(n_smes_u*8):.2%}")
        logger.info("═" * 60)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_realism(self) -> bool:
        logger.info("═" * 60)
        logger.info("REALISM VALIDATION")
        passed = 0
        total  = 0

        def chk(ok, label, detail=""):
            nonlocal passed, total
            total += 1
            if ok: passed += 1
            flag = "PASS" if ok else "FAIL"
            logger.info(f"  [{flag}] {label}" + (f" — {detail}" if detail else ""))

        # 1. Revenue log-mean in expected USD range
        rev = self.sme_df["annual_revenue_usd"].dropna()
        lm  = float(np.log(rev[rev > 0]).mean())
        chk(5.5 < lm < 9.0, "Revenue log-mean in expected range", f"got {lm:.2f}")

        # 2. Default rate ≈ UCI 22%
        nl = self.fin_df["n_previous_loans"].sum()
        nd = self.fin_df["n_defaults"].clip(0, self.fin_df["n_previous_loans"]).sum()
        dr = nd / max(nl, 1)
        chk(0.12 < dr < 0.35, "Default rate matches UCI (~22%)", f"got {dr:.3f}")

        # 3. Mobile money rate realistic
        mm = (self.fin_df["has_mobile_money"] == "Have now").mean()
        chk(0.40 < mm < 0.85, "Mobile money in realistic range", f"got {mm:.3f}")

        # 4. Matrix sparsity 75-95%
        n_p = self.inter_df.groupby(["sme_id", "product_id"]).ngroups
        n_s = self.sme_df["sme_id"].nunique()
        sp  = 1 - n_p / (n_s * 8)
        chk(0.70 < sp < 0.97, "Matrix sparsity 70–97%", f"got {sp:.2%}")

        # 5. No SME has > 8 products
        per_sme = self.inter_df.groupby("sme_id")["product_id"].nunique()
        chk((per_sme <= 8).all(), "No SME has > 8 products", f"max={per_sme.max()}")

        # 6. Most interaction dates after registration
        m = self.inter_df.merge(
            self.sme_df[["sme_id", "registration_date"]], on="sme_id", how="left"
        )
        m["registration_date"] = pd.to_datetime(m["registration_date"])
        m["interaction_date"]  = pd.to_datetime(m["interaction_date"])
        pct_before = (m["interaction_date"] < m["registration_date"]).mean()
        chk(pct_before < 0.05, "Interaction dates mostly after registration",
            f"{pct_before:.2%} before")

        # 7. Business age mean realistic
        biz_age = self.sme_df["years_in_business"].dropna().mean()
        chk(2 < biz_age < 12, "Business age mean realistic", f"got {biz_age:.1f} yr")

        # 8. Country distribution matches target weights
        country_dist = self.sme_df["country"].value_counts(normalize=True)
        kenya_share  = country_dist.get("Kenya", 0)
        chk(0.15 < kenya_share < 0.30, "Kenya share ~22%", f"got {kenya_share:.3f}")

        logger.info(f"\n  Result: {passed}/{total} checks passed")
        logger.info("═" * 60)
        return passed == total


# ── Helpers ────────────────────────────────────────────────────────────────────

def _isnan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return v is None


def _irow(sme_id, prod_id, itype, date, amt_req, amt_app, sat, rep, reason, channel) -> dict:
    return {
        "sme_id":               sme_id,
        "product_id":           prod_id,
        "interaction_type":     itype,
        "interaction_date":     date,
        "amount_requested_usd": round(float(amt_req), 2) if not _isnan(amt_req) else np.nan,
        "amount_approved_usd":  round(float(amt_app), 2) if not _isnan(amt_app) else np.nan,
        "satisfaction_score":   int(sat) if not _isnan(sat) else np.nan,
        "repayment_rate_pct":   round(float(rep), 1) if not _isnan(rep) else np.nan,
        "reason_rejection":     reason,
        "channel":              channel,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen  = RawDataGenerator(data_raw_path="data/raw", random_state=42)
    data = gen.generate_all(n_smes=5000)
    gen.validate_realism()
