# IAN(inflammation-nutrition score)predictionCKDofNHANESdata analysis report

**analysis time**: 2025-06-16

**data source**: NHANES 2011-2018 (4cycles)

**analysis population**: adult (age ≥ 20years)

**CKDdefinition**: eGFR < 60 mL/min/1.73m² or UACR ≥ 30 mg/g

---

## abstract

- total sample size: 20,222 persons
- validIAN data: 20,222 persons
- CKDpatient: 3,703 persons (18.3%)
- nonCKD: 16,519 persons (81.7%)

## IAN score construction method

IAN(Inflammation-Nutrition Score,inflammation-nutrition score)by3components were used to build:

| indicator | meaning | scoring rule |
|------|------|----------|
| NLR(neutrophil/lymphocyte ratio) | systemic inflammation marker | T1=0, T2=1, T3=2 (higher is worse) |
| hemoglobin (Hb) | anemia/nutritional status | T1=2, T2=1, T3=0 (lower is worse,inverse) |
| albumin | nutritional status | T1=2, T2=1, T3=0 (lower is worse,inverse) |

**total score range**: 0-6point (higher indicates worse inflammation/worse nutritional status)

**grade**: Low risk (0-2), Medium risk (3-4), High risk (5-6)

## IAN score andCKDassociation

### by IANstratified by score CKD prevalence

|   IAN score |   total count |   CKDcount |   CKDprevalence(%) |
|------------:|--------------:|-----------:|-------------------:|
|           0 |           848 |         65 |            7.66509 |
|           1 |          2316 |        207 |            8.93782 |
|           2 |          3911 |        506 |           12.9379  |
|           3 |          4493 |        727 |           16.1807  |
|           4 |          4509 |        954 |           21.1577  |
|           5 |          2691 |        682 |           25.3437  |
|           6 |          1454 |        562 |           38.652   |

### by IANstratified by gradeCKDprevalence

| IAN grade         |   total count |   CKDcount |   CKDprevalence(%) |
|:------------------|--------------:|-----------:|-------------------:|
| Low risk (0-2)    |          7075 |        778 |            10.9965 |
| Medium risk (3-4) |          9002 |       1681 |            18.6736 |
| High risk (5-6)   |          4145 |       1244 |            30.0121 |

## Logistic regression: IANpredictionCKD

### model1: IAN score (continuous) — univariate

- OR = 1.394 (per 1-point increase1point)
- AUC = 0.6372

### model2: IAN grade — univariate

- IAN Medium risk  (vs low risk): OR=1.86 (95%CI: 1.70-2.04), P=<0.001
- IAN High risk  (vs low risk): OR=3.47 (95%CI: 3.14-3.84), P=<0.001
- AUC = 0.6201

### model3: IAN + age + sex + race (multivariate adjustment)

- IAN score(per 1-point increment): OR = 1.282
- after adjustmentIANstill significantly predictsCKD
- AUC = 0.7666

## key findings

1. **IAN score and CKD are significantly correlated**: For each 1-point increase in IAN score, CKD risk increases by 39.4% (OR=1.394).

2. **dose-response relationship**: As IAN score increases from 0 to 6 points, CKD prevalence shows an increasing trend, demonstrating a clear dose-response relationship.

3. **IAN component contribution**: NLR, low hemoglobin and low albumin are each independently associated with CKD risk; the combination of the three components yields a stronger predictive performance.

4. **independent predictive value**: After adjusting for age, sex and race, IAN score still significantly predicts CKD, suggesting it has predictive value independent of traditional risk factors.

5. **clinical significance**: The IAN score is a composite indicator based on routine laboratory tests; it is simple to operate, low cost, and suitable for CKD risk stratification in primary care or large-scale screening.

## visualization figures

| figure number | description | filename |
|------|------|--------|
| figure1 | IAN score distribution and gradingCKDprevalence | fig1_ian_distribution.png |
| figure2 | IANcomponent forest plot | fig2_forest_ian_components.png |
| figure3 | ROC curve | fig3_roc_curve.png |
| figure4 | IANandCKDprevalence trend | fig4_ian_trend.png |
| figure5 | IANcomponent boxplots | fig5_boxplots.png |
| figure6 | correlation heatmap | fig6_correlation_heatmap.png |

