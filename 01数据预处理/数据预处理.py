import pandas as pd
import os
from pathlib import Path
input_file="整合转置数据_完整.csv"
# 使用脚本所在目录作为基准路径，避免工作目录不同导致找不到文件
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / input_file,
                 encoding='gbk',
                 header=0,   # 第 0 行作为列名
                 index_col=0)  # 第 0 列作为行索引
df = df.dropna(axis=1, how='all')
df = df.astype('float32')
#输出df的维度
print(df.shape)


#请对df中列值全为一个值的列进行删除
df = df.loc[:, (df != df.iloc[0]).any()]
#输出df的维度
print(df.shape)

# 保留每类前缀的前10个列：疾病/药物/检验
prefix_targets = [('疾病', 200), ('药物', 200), ('检验', 200)]
selected_cols = []
for prefix, k in prefix_targets:
    matched = [c for c in df.columns if str(c).startswith(prefix)]
    selected_cols.extend(matched[:k])
# 去重保持原始顺序
selected_cols = list(dict.fromkeys(selected_cols))
# 只保留选中的列
df = df.loc[:, selected_cols]
print('按前缀筛选后的维度:', df.shape)

from sklearn.impute import SimpleImputer
import pandas as pd

# 1. 用众数填充
imputer = SimpleImputer(strategy='most_frequent')   # 每列分别找出现最多的 0 或 1
imputed_array = imputer.fit_transform(df)

# 2. 还原成 DataFrame
df_imputed = pd.DataFrame(imputed_array,
                          columns=df.columns,
                          index=df.index)

print("\n众数插补后的数据框（df_imputed）：")
print(df_imputed.head())

# 清洗列名：删除 [] 及其中内容，规范空白并唯一化
df_imputed.columns = (
    df_imputed.columns.astype(str)
    .str.replace(r"\[.*?\]", "", regex=True)  # 移除方括号及其中内容
    .str.strip()
)

_seen = {}
_clean_cols = []
for col in df_imputed.columns:
    base = col
    if base in _seen:
        _seen[base] += 1
        _clean_cols.append(f"{base}_{_seen[base]}")  # 重名加后缀确保唯一
    else:
        _seen[base] = 0
        _clean_cols.append(base)
df_imputed.columns = _clean_cols

df_imputed = df_imputed.iloc[:6000, :]
#输出df_imputed的维度
print(df_imputed.shape)
print(df_imputed.head())
#将df_imputed保存到文件中（保存到脚本同目录）
df_imputed.to_csv(BASE_DIR / '缩减数据_规格.csv', index=True)