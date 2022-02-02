---
layout: single
title: "3분의 결과를 이용해 다음 매매를 결정하는 3분 매매법"
categories: Trading
tag: [python, upbit]
---

# 3분 매매법
3분 양봉이 나오면 다음 1분 시작시에 매수하고 1분이 끝날 때 매도하는 방법

![chart](/assets/images/2022-02-03-upbit-btc.jpg)


**Github**

[NoteBook](https://github.com/helpingstar/hstrader/blob/main/three_minute_momentum.ipynb)

[upbit_quotation.py](https://github.com/helpingstar/hstrader/blob/main/upbit_quotation.py)

업비트 비트코인 1분 차트를 보던 중 3분동안 양봉이 나온다면 (양봉이 3개라면) 다음 1분동안 양봉이 나올 확률이 있지 않을까? 해서 해본 백테스트.




```python
from upbit_quotation import *
FEE = 0.0005
```


`from` 시간부터 `to(default=now)` 시간까지의 `unit` 분봉 `DataFrame`을 얻는다.

```python
df_2022 = get_minute_candle_from_to(1, 'KRW-BTC', '2022-01-01T00:00:00')
```


```python
df_2022
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>market</th>
      <th>candle_date_time_utc</th>
      <th>candle_date_time_kst</th>
      <th>opening_price</th>
      <th>high_price</th>
      <th>low_price</th>
      <th>trade_price</th>
      <th>timestamp</th>
      <th>candle_acc_trade_price</th>
      <th>candle_acc_trade_volume</th>
      <th>unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:05:00</td>
      <td>2022-02-03T04:05:00</td>
      <td>46457000</td>
      <td>46482000</td>
      <td>46451000</td>
      <td>46451000</td>
      <td>2022-02-02 19:05:28.113</td>
      <td>1.384628e+07</td>
      <td>0.297958</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:04:00</td>
      <td>2022-02-03T04:04:00</td>
      <td>46451000</td>
      <td>46462000</td>
      <td>46451000</td>
      <td>46462000</td>
      <td>2022-02-02 19:04:58.756</td>
      <td>2.057519e+07</td>
      <td>0.442904</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:03:00</td>
      <td>2022-02-03T04:03:00</td>
      <td>46439000</td>
      <td>46452000</td>
      <td>46421000</td>
      <td>46451000</td>
      <td>2022-02-02 19:03:59.589</td>
      <td>1.549428e+07</td>
      <td>0.333660</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:02:00</td>
      <td>2022-02-03T04:02:00</td>
      <td>46439000</td>
      <td>46439000</td>
      <td>46408000</td>
      <td>46439000</td>
      <td>2022-02-02 19:02:57.967</td>
      <td>1.083041e+08</td>
      <td>2.333458</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:01:00</td>
      <td>2022-02-03T04:01:00</td>
      <td>46415000</td>
      <td>46447000</td>
      <td>46415000</td>
      <td>46439000</td>
      <td>2022-02-02 19:01:56.361</td>
      <td>3.910157e+07</td>
      <td>0.842121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47524</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:04:00</td>
      <td>2022-01-01T00:04:00</td>
      <td>58441000</td>
      <td>58445000</td>
      <td>58405000</td>
      <td>58435000</td>
      <td>2021-12-31 15:04:53.917</td>
      <td>1.304826e+08</td>
      <td>2.233001</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47525</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:03:00</td>
      <td>2022-01-01T00:03:00</td>
      <td>58423000</td>
      <td>58442000</td>
      <td>58422000</td>
      <td>58441000</td>
      <td>2021-12-31 15:03:59.557</td>
      <td>1.007087e+08</td>
      <td>1.723417</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47526</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:02:00</td>
      <td>2022-01-01T00:02:00</td>
      <td>58480000</td>
      <td>58480000</td>
      <td>58405000</td>
      <td>58441000</td>
      <td>2021-12-31 15:02:58.349</td>
      <td>9.058170e+07</td>
      <td>1.550462</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47527</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:01:00</td>
      <td>2022-01-01T00:01:00</td>
      <td>58462000</td>
      <td>58485000</td>
      <td>58421000</td>
      <td>58480000</td>
      <td>2021-12-31 15:02:00.101</td>
      <td>3.546750e+08</td>
      <td>6.065133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47528</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:00:00</td>
      <td>2022-01-01T00:00:00</td>
      <td>58412000</td>
      <td>58477000</td>
      <td>58405000</td>
      <td>58471000</td>
      <td>2021-12-31 15:01:00.079</td>
      <td>1.751670e+08</td>
      <td>2.996578</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>47529 rows × 11 columns</p>
</div>



상승이면 `1`, 하강이면 `-1`을 기입하는 Column을 새로 만든다


```python
df_2022['move'] = np.where(df_2022['opening_price'] < df_2022['trade_price'], 1, -1)
```


```python
df_2022
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>market</th>
      <th>candle_date_time_utc</th>
      <th>candle_date_time_kst</th>
      <th>opening_price</th>
      <th>high_price</th>
      <th>low_price</th>
      <th>trade_price</th>
      <th>timestamp</th>
      <th>candle_acc_trade_price</th>
      <th>candle_acc_trade_volume</th>
      <th>unit</th>
      <th>move</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:05:00</td>
      <td>2022-02-03T04:05:00</td>
      <td>46457000</td>
      <td>46482000</td>
      <td>46451000</td>
      <td>46451000</td>
      <td>2022-02-02 19:05:28.113</td>
      <td>1.384628e+07</td>
      <td>0.297958</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:04:00</td>
      <td>2022-02-03T04:04:00</td>
      <td>46451000</td>
      <td>46462000</td>
      <td>46451000</td>
      <td>46462000</td>
      <td>2022-02-02 19:04:58.756</td>
      <td>2.057519e+07</td>
      <td>0.442904</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:03:00</td>
      <td>2022-02-03T04:03:00</td>
      <td>46439000</td>
      <td>46452000</td>
      <td>46421000</td>
      <td>46451000</td>
      <td>2022-02-02 19:03:59.589</td>
      <td>1.549428e+07</td>
      <td>0.333660</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:02:00</td>
      <td>2022-02-03T04:02:00</td>
      <td>46439000</td>
      <td>46439000</td>
      <td>46408000</td>
      <td>46439000</td>
      <td>2022-02-02 19:02:57.967</td>
      <td>1.083041e+08</td>
      <td>2.333458</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:01:00</td>
      <td>2022-02-03T04:01:00</td>
      <td>46415000</td>
      <td>46447000</td>
      <td>46415000</td>
      <td>46439000</td>
      <td>2022-02-02 19:01:56.361</td>
      <td>3.910157e+07</td>
      <td>0.842121</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47524</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:04:00</td>
      <td>2022-01-01T00:04:00</td>
      <td>58441000</td>
      <td>58445000</td>
      <td>58405000</td>
      <td>58435000</td>
      <td>2021-12-31 15:04:53.917</td>
      <td>1.304826e+08</td>
      <td>2.233001</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>47525</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:03:00</td>
      <td>2022-01-01T00:03:00</td>
      <td>58423000</td>
      <td>58442000</td>
      <td>58422000</td>
      <td>58441000</td>
      <td>2021-12-31 15:03:59.557</td>
      <td>1.007087e+08</td>
      <td>1.723417</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47526</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:02:00</td>
      <td>2022-01-01T00:02:00</td>
      <td>58480000</td>
      <td>58480000</td>
      <td>58405000</td>
      <td>58441000</td>
      <td>2021-12-31 15:02:58.349</td>
      <td>9.058170e+07</td>
      <td>1.550462</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>47527</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:01:00</td>
      <td>2022-01-01T00:01:00</td>
      <td>58462000</td>
      <td>58485000</td>
      <td>58421000</td>
      <td>58480000</td>
      <td>2021-12-31 15:02:00.101</td>
      <td>3.546750e+08</td>
      <td>6.065133</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47528</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:00:00</td>
      <td>2022-01-01T00:00:00</td>
      <td>58412000</td>
      <td>58477000</td>
      <td>58405000</td>
      <td>58471000</td>
      <td>2021-12-31 15:01:00.079</td>
      <td>1.751670e+08</td>
      <td>2.996578</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>47529 rows × 12 columns</p>
</div>



`iterrow()`는 위에서부터 진행하기 때문에 `DataFrame`을 뒤집는다.


```python
df_2022 = df_2022[::-1]
```


```python
df_2022
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>market</th>
      <th>candle_date_time_utc</th>
      <th>candle_date_time_kst</th>
      <th>opening_price</th>
      <th>high_price</th>
      <th>low_price</th>
      <th>trade_price</th>
      <th>timestamp</th>
      <th>candle_acc_trade_price</th>
      <th>candle_acc_trade_volume</th>
      <th>unit</th>
      <th>move</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47528</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:00:00</td>
      <td>2022-01-01T00:00:00</td>
      <td>58412000</td>
      <td>58477000</td>
      <td>58405000</td>
      <td>58471000</td>
      <td>2021-12-31 15:01:00.079</td>
      <td>1.751670e+08</td>
      <td>2.996578</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47527</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:01:00</td>
      <td>2022-01-01T00:01:00</td>
      <td>58462000</td>
      <td>58485000</td>
      <td>58421000</td>
      <td>58480000</td>
      <td>2021-12-31 15:02:00.101</td>
      <td>3.546750e+08</td>
      <td>6.065133</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47526</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:02:00</td>
      <td>2022-01-01T00:02:00</td>
      <td>58480000</td>
      <td>58480000</td>
      <td>58405000</td>
      <td>58441000</td>
      <td>2021-12-31 15:02:58.349</td>
      <td>9.058170e+07</td>
      <td>1.550462</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>47525</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:03:00</td>
      <td>2022-01-01T00:03:00</td>
      <td>58423000</td>
      <td>58442000</td>
      <td>58422000</td>
      <td>58441000</td>
      <td>2021-12-31 15:03:59.557</td>
      <td>1.007087e+08</td>
      <td>1.723417</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47524</th>
      <td>KRW-BTC</td>
      <td>2021-12-31T15:04:00</td>
      <td>2022-01-01T00:04:00</td>
      <td>58441000</td>
      <td>58445000</td>
      <td>58405000</td>
      <td>58435000</td>
      <td>2021-12-31 15:04:53.917</td>
      <td>1.304826e+08</td>
      <td>2.233001</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:01:00</td>
      <td>2022-02-03T04:01:00</td>
      <td>46415000</td>
      <td>46447000</td>
      <td>46415000</td>
      <td>46439000</td>
      <td>2022-02-02 19:01:56.361</td>
      <td>3.910157e+07</td>
      <td>0.842121</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:02:00</td>
      <td>2022-02-03T04:02:00</td>
      <td>46439000</td>
      <td>46439000</td>
      <td>46408000</td>
      <td>46439000</td>
      <td>2022-02-02 19:02:57.967</td>
      <td>1.083041e+08</td>
      <td>2.333458</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:03:00</td>
      <td>2022-02-03T04:03:00</td>
      <td>46439000</td>
      <td>46452000</td>
      <td>46421000</td>
      <td>46451000</td>
      <td>2022-02-02 19:03:59.589</td>
      <td>1.549428e+07</td>
      <td>0.333660</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:04:00</td>
      <td>2022-02-03T04:04:00</td>
      <td>46451000</td>
      <td>46462000</td>
      <td>46451000</td>
      <td>46462000</td>
      <td>2022-02-02 19:04:58.756</td>
      <td>2.057519e+07</td>
      <td>0.442904</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>KRW-BTC</td>
      <td>2022-02-02T19:05:00</td>
      <td>2022-02-03T04:05:00</td>
      <td>46457000</td>
      <td>46482000</td>
      <td>46451000</td>
      <td>46451000</td>
      <td>2022-02-02 19:05:28.113</td>
      <td>1.384628e+07</td>
      <td>0.297958</td>
      <td>1</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>47529 rows × 12 columns</p>
</div>



예산은 `10000`으로 설정하고 `count`가 `3`이 되면 다음 `row`는 무조건 시가에 매수하고 종가에 매도한다, 그리고 `count`는 0으로 만든 후 `continue` 한다


```python
budget = 10000
count = 0
rise_count = 0
fall_count = 0
result_budget = budget

for _, member in df_2022.iterrows():
    if count == 3:
        result_budget *= 1 - FEE
        result_budget *= (member['trade_price'] / member['opening_price'])
        result_budget *= 1 - FEE
        if member['move'] == 1:
            rise_count += 1
        else:
            fall_count += 1
        count = 0
        continue
    if member['move'] == 1:
        count += 1
    if member['move'] == -1:
        count = 0
```

수익률 **-93.14%**, 무조건 망하니 생각도 하지 말자


```python
pd.DataFrame({'year': [2022], 'budget': [result_budget], 'rise_count': [rise_count], 'fall_count': [fall_count], 'earnings rate': [(result_budget / budget) - 1]}).set_index('year')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>rise_count</th>
      <th>fall_count</th>
      <th>earnings rate</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022</th>
      <td>682.448528</td>
      <td>1373</td>
      <td>1418</td>
      <td>-0.931755</td>
    </tr>
  </tbody>
</table>
</div>



위의 과정을 함수화 하여 다른 연도에도 적용해보자


```python
df_2021 = get_minute_candle_from_to(1, 'KRW-BTC', start='2021-01-01T00:00:00', end='2022-01-01T00:00:00')
df_2020 = get_minute_candle_from_to(1, 'KRW-BTC', start='2020-01-01T00:00:00', end='2021-01-01T00:00:00')
```


```python
def m3_strategy_info(df, budget, year):
    df['move'] = np.where(df['opening_price'] < df['trade_price'], 1, -1)
    df = df[::-1]
    count = 0
    rise_count = 0
    fall_count = 0
    result_budget = budget

    for _, member in df.iterrows():
        if count == 3:
            result_budget *= 1 - FEE
            result_budget *= (member['trade_price'] / member['opening_price'])
            result_budget *= 1 - FEE
            if member['move'] == 1:
                rise_count += 1
            else:
                fall_count += 1
            count = 0
            continue
        if member['move'] == 1:
            count += 1
        if member['move'] == -1:
            count = 0
    return pd.DataFrame({'year': [year], 'budget': [result_budget], 'rise_count': [rise_count], 'fall_count': [fall_count], 'earnings rate': [(result_budget / budget) - 1]}).set_index('year')
```


```python
m3_strategy_info(df_2021, 10000, 2021)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>rise_count</th>
      <th>fall_count</th>
      <th>earnings rate</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021</th>
      <td>3.421880e-10</td>
      <td>17191</td>
      <td>15832</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
m3_strategy_info(df_2020, 10000, 2020)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>rise_count</th>
      <th>fall_count</th>
      <th>earnings rate</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020</th>
      <td>0.000017</td>
      <td>9754</td>
      <td>11558</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>



각각 수익률 **-100%** 를 달성하였다. 말도 안되는 전략이었던 것 같다.
