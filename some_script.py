import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================
df = pd.read_parquet('transaction_fraud_data.parquet')
exchange_rates = pd.read_parquet('historical_currency_exchange.parquet')

print("="*60)
print("ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("="*60)
print(f"Количество транзакций: {len(df):,}")
print(f"Количество клиентов: {df['customer_id'].nunique():,}")
print(f"Период данных: {df['timestamp'].min()} — {df['timestamp'].max()}")
print(f"\nРазмер датасета: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================
# АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (FRAUD)
# ============================================
print("\n" + "="*60)
print("АНАЛИЗ МОШЕННИЧЕСТВА")
print("="*60)

fraud_stats = df['is_fraud'].value_counts()
fraud_rate = df['is_fraud'].mean() * 100
print(f"Легитимных транзакций: {fraud_stats[False]:,} ({100-fraud_rate:.2f}%)")
print(f"Мошеннических транзакций: {fraud_stats[True]:,} ({fraud_rate:.2f}%)")

# ============================================
# ИЗВЛЕЧЕНИЕ ВЛОЖЕННЫХ ПОЛЕЙ
# ============================================
if isinstance(df['last_hour_activity'].iloc[0], dict):
    df['lha_num_transactions'] = df['last_hour_activity'].apply(lambda x: x['num_transactions'])
    df['lha_total_amount'] = df['last_hour_activity'].apply(lambda x: x['total_amount'])
    df['lha_unique_merchants'] = df['last_hour_activity'].apply(lambda x: x['unique_merchants'])
    df['lha_unique_countries'] = df['last_hour_activity'].apply(lambda x: x['unique_countries'])
    df['lha_max_single_amount'] = df['last_hour_activity'].apply(lambda x: x['max_single_amount'])
else:
    df['lha_num_transactions'] = df['last_hour_activity'].struct.field('num_transactions')
    df['lha_total_amount'] = df['last_hour_activity'].struct.field('total_amount')
    df['lha_unique_merchants'] = df['last_hour_activity'].struct.field('unique_merchants')
    df['lha_unique_countries'] = df['last_hour_activity'].struct.field('unique_countries')
    df['lha_max_single_amount'] = df['last_hour_activity'].struct.field('max_single_amount')

# ============================================
# АНАЛИЗ ПО КАТЕГОРИЯМ
# ============================================
print("\n" + "="*60)
print("FRAUD RATE ПО КАТЕГОРИЯМ")
print("="*60)

# По каналам
print("\n--- По каналу (channel) ---")
channel_fraud = df.groupby('channel')['is_fraud'].agg(['sum', 'count', 'mean'])
channel_fraud.columns = ['fraud_count', 'total', 'fraud_rate']
channel_fraud['fraud_rate'] = (channel_fraud['fraud_rate'] * 100).round(2)
print(channel_fraud.sort_values('fraud_rate', ascending=False))

# По типу карты
print("\n--- По типу карты (card_type) ---")
card_fraud = df.groupby('card_type')['is_fraud'].agg(['sum', 'count', 'mean'])
card_fraud.columns = ['fraud_count', 'total', 'fraud_rate']
card_fraud['fraud_rate'] = (card_fraud['fraud_rate'] * 100).round(2)
print(card_fraud.sort_values('fraud_rate', ascending=False))

# По категории вендора
print("\n--- По категории вендора (vendor_category) ---")
vendor_fraud = df.groupby('vendor_category')['is_fraud'].agg(['sum', 'count', 'mean'])
vendor_fraud.columns = ['fraud_count', 'total', 'fraud_rate']
vendor_fraud['fraud_rate'] = (vendor_fraud['fraud_rate'] * 100).round(2)
print(vendor_fraud.sort_values('fraud_rate', ascending=False))

# По странам (топ-10)
print("\n--- Топ-10 стран по количеству фрода ---")
country_fraud = df.groupby('country')['is_fraud'].agg(['sum', 'count', 'mean'])
country_fraud.columns = ['fraud_count', 'total', 'fraud_rate']
country_fraud['fraud_rate'] = (country_fraud['fraud_rate'] * 100).round(2)
print(country_fraud.sort_values('fraud_count', ascending=False).head(10))

# ============================================
# АНАЛИЗ БУЛЕВЫХ ПРИЗНАКОВ
# ============================================
print("\n" + "="*60)
print("ВЛИЯНИЕ БУЛЕВЫХ ПРИЗНАКОВ НА FRAUD")
print("="*60)

bool_features = ['is_card_present', 'is_outside_home_country', 'is_high_risk_vendor', 'is_weekend']
for feat in bool_features:
    fraud_when_true = df[df[feat] == True]['is_fraud'].mean() * 100
    fraud_when_false = df[df[feat] == False]['is_fraud'].mean() * 100
    print(f"{feat}:")
    print(f"  True:  {fraud_when_true:.2f}%")
    print(f"  False: {fraud_when_false:.2f}%")
    print(f"  Разница: {abs(fraud_when_true - fraud_when_false):.2f}%\n")

# ============================================
# АНАЛИЗ СУММ ТРАНЗАКЦИЙ
# ============================================
print("\n" + "="*60)
print("АНАЛИЗ СУММ ТРАНЗАКЦИЙ")
print("="*60)

print("\n--- Статистика по сумме (amount) ---")
print(df.groupby('is_fraud')['amount'].describe())

print("\n--- Средняя сумма по fraud/non-fraud ---")
print(f"Легитимные: {df[df['is_fraud']==False]['amount'].mean():.2f}")
print(f"Мошеннические: {df[df['is_fraud']==True]['amount'].mean():.2f}")

# ============================================
# АНАЛИЗ АКТИВНОСТИ ЗА ПОСЛЕДНИЙ ЧАС
# ============================================
print("\n" + "="*60)
print("АКТИВНОСТЬ ЗА ПОСЛЕДНИЙ ЧАС (last_hour_activity)")
print("="*60)

lha_features = ['lha_num_transactions', 'lha_total_amount', 'lha_unique_merchants', 'lha_unique_countries']
for feat in lha_features:
    fraud_mean = df[df['is_fraud']==True][feat].mean()
    legit_mean = df[df['is_fraud']==False][feat].mean()
    print(f"{feat}:")
    print(f"  Fraud: {fraud_mean:.2f}")
    print(f"  Legit: {legit_mean:.2f}")
    print(f"  Ratio: {fraud_mean/legit_mean:.2f}x\n")

# ============================================
# ВРЕМЕННОЙ АНАЛИЗ
# ============================================
print("\n" + "="*60)
print("ВРЕМЕННОЙ АНАЛИЗ")
print("="*60)

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

print("\n--- Fraud rate по часам ---")
hourly_fraud = df.groupby('hour')['is_fraud'].mean() * 100
print(hourly_fraud.sort_values(ascending=False).head(5))

print("\n--- Fraud rate по дням недели ---")
daily_fraud = df.groupby('day_of_week')['is_fraud'].mean() * 100
days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
for i, rate in daily_fraud.items():
    print(f"  {days[i]}: {rate:.2f}%")

# ============================================
# АНАЛИЗ УСТРОЙСТВ
# ============================================
print("\n" + "="*60)
print("АНАЛИЗ УСТРОЙСТВ")
print("="*60)

device_fraud = df.groupby('device')['is_fraud'].agg(['sum', 'count', 'mean'])
device_fraud.columns = ['fraud_count', 'total', 'fraud_rate']
device_fraud['fraud_rate'] = (device_fraud['fraud_rate'] * 100).round(2)
print(device_fraud.sort_values('fraud_rate', ascending=False))

# ============================================
# КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ============================================
print("\n" + "="*60)
print("КОРРЕЛЯЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ С FRAUD")
print("="*60)

numeric_cols = ['amount', 'lha_num_transactions', 'lha_total_amount', 
                'lha_unique_merchants', 'lha_unique_countries', 'lha_max_single_amount']
correlations = df[numeric_cols + ['is_fraud']].corr()['is_fraud'].drop('is_fraud').sort_values(key=abs, ascending=False)
print(correlations)

# ============================================
# СОХРАНЕНИЕ ВИЗУАЛИЗАЦИЙ
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Fraud rate по каналам
ax1 = axes[0, 0]
channel_fraud_plot = df.groupby('channel')['is_fraud'].mean() * 100
channel_fraud_plot.plot(kind='bar', ax=ax1, color='coral')
ax1.set_title('Fraud Rate по каналам')
ax1.set_ylabel('Fraud Rate (%)')
ax1.tick_params(axis='x', rotation=45)

# 2. Fraud rate по часам
ax2 = axes[0, 1]
hourly_fraud.plot(kind='line', ax=ax2, marker='o', color='steelblue')
ax2.set_title('Fraud Rate по часам суток')
ax2.set_xlabel('Час')
ax2.set_ylabel('Fraud Rate (%)')

# 3. Распределение сумм
ax3 = axes[1, 0]
df[df['is_fraud']==False]['amount'].hist(bins=50, ax=ax3, alpha=0.7, label='Legit', color='green')
df[df['is_fraud']==True]['amount'].hist(bins=50, ax=ax3, alpha=0.7, label='Fraud', color='red')
ax3.set_title('Распределение сумм транзакций')
ax3.set_xlabel('Amount')
ax3.legend()
ax3.set_xlim(0, df['amount'].quantile(0.95))

# 4. Fraud rate по категориям вендора
ax4 = axes[1, 1]
vendor_fraud_plot = df.groupby('vendor_category')['is_fraud'].mean() * 100
vendor_fraud_plot.sort_values().plot(kind='barh', ax=ax4, color='purple')
ax4.set_title('Fraud Rate по категориям вендора')
ax4.set_xlabel('Fraud Rate (%)')

plt.tight_layout()
plt.savefig('eda_fraud_analysis.png', dpi=150)
plt.show()

print("\n✅ Визуализация сохранена в 'eda_fraud_analysis.png'")
