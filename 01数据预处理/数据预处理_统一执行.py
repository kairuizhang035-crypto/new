import os
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PREPROCESS_FILE = Path('/home/zkr/因果发现3/01数据预处理/缩减数据_规格.csv')
REPORT_FILE = Path('/home/zkr/因果发现3/01数据预处理/数据处理报告.md')

def read_csv(fp: Path):
    encs = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']
    for e in encs:
        try:
            return pd.read_csv(fp, encoding=e, header=0, index_col=0)
        except Exception:
            try:
                return pd.read_csv(fp, encoding=e, header=0)
            except Exception:
                continue
    return pd.read_csv(fp)

def is_binary_series(s: pd.Series):
    vals = pd.unique(s.dropna())
    return len(vals) <= 2

def build_report(before_df: pd.DataFrame, after_df: pd.DataFrame, methods: list, notes: list):
    def col_missing(df):
        return df.isna().sum().to_dict()
    def dtype_summary(df):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in num_cols]
        return {
            'numeric_count': len(num_cols),
            'categorical_count': len(cat_cols)
        }
    b = {
        'shape': list(before_df.shape),
        'missing_by_column': col_missing(before_df),
        'dtypes': dtype_summary(before_df)
    }
    a = {
        'shape': list(after_df.shape),
        'missing_by_column': col_missing(after_df),
        'dtypes': dtype_summary(after_df)
    }
    lines = []
    lines.append('# 数据处理报告')
    lines.append('')
    lines.append('## 处理前统计')
    lines.append(f"- 维度: {b['shape'][0]} × {b['shape'][1]}")
    lines.append(f"- 数值列数: {b['dtypes']['numeric_count']}")
    lines.append(f"- 类别列数: {b['dtypes']['categorical_count']}")
    lines.append('- 缺失值按列汇总:')
    lines.append(str(b['missing_by_column']))
    lines.append('')
    lines.append('## 处理后统计')
    lines.append(f"- 维度: {a['shape'][0]} × {a['shape'][1]}")
    lines.append(f"- 数值列数: {a['dtypes']['numeric_count']}")
    lines.append(f"- 类别列数: {a['dtypes']['categorical_count']}")
    lines.append('- 缺失值按列汇总:')
    lines.append(str(a['missing_by_column']))
    lines.append('')
    lines.append('## 处理方法')
    for m in methods:
        lines.append(f"- {m}")
    lines.append('')
    lines.append('## 注意事项')
    for n in notes:
        lines.append(f"- {n}")
    REPORT_FILE.write_text('\n'.join(lines), encoding='utf-8')

def main():
    df0 = read_csv(PREPROCESS_FILE)
    df = df0.copy()
    methods = []
    notes = []

    df.columns = (
        df.columns.astype(str)
        .str.replace(r"\[.*?\]", "", regex=True)
        .str.strip()
    )
    seen = {}
    clean_cols = []
    for col in df.columns:
        base = col
        if base in seen:
            seen[base] += 1
            clean_cols.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            clean_cols.append(base)
    df.columns = clean_cols
    methods.append('列名清洗与去重唯一化')

    num_cols = []
    cat_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() >= max(1, int(0.7 * len(s))):
            df[c] = s
            num_cols.append(c)
        else:
            df[c] = df[c].astype(str)
            cat_cols.append(c)

    if cat_cols:
        imp_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
        methods.append('类别列众数插补')
    if num_cols:
        imp_num = SimpleImputer(strategy='median')
        df[num_cols] = imp_num.fit_transform(df[num_cols])
        methods.append('数值列中位数插补')
    if num_cols:
        qlow = df[num_cols].quantile(0.01)
        qhigh = df[num_cols].quantile(0.99)
        df[num_cols] = df[num_cols].clip(lower=qlow, upper=qhigh, axis=1)
        methods.append('数值列分位剪裁异常值')

    bin_cols = [c for c in num_cols if is_binary_series(df[c])]
    scale_cols = [c for c in num_cols if c not in bin_cols]
    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        methods.append('数值列标准化')

    for c in cat_cols:
        codes, _ = pd.factorize(df[c])
        df[c] = codes
    methods.append('类别列标签编码')

    notes.append('未删除任何行或列，保持原始规模不变')
    notes.append('标准化不会改变数据规模，但会改变值的量纲')
    notes.append('异常值剪裁以分位数为界，避免极端值影响')

    df.to_csv(PREPROCESS_FILE, index=True, encoding='utf-8-sig')
    build_report(df0, df, methods, notes)

if __name__ == '__main__':
    main()
