# Dataset audit

Directory: `data\synthetic`

## Size

- labelled pages: **420**
- images with no label file: **0**
- writers: **14**
- clients: **336**
- layouts: two_column (115), label_value (106), narrative (101), printed_form (98)
- fields per page: min 10, mean 20.8, max 38

## Duplicates and unreadable pages

- exact duplicate groups: **0** (0 redundant pages)
- near-duplicate pairs: **39**
- unreadable images: **0**

## Personal data

- label files containing personal data: **385** of 420
  - email: 312 occurrence(s)
  - ni_number: 78 occurrence(s)
  - uk_phone: 360 occurrence(s)

> These values stay out of git (`data/` is ignored) and out of every report. If any of this is real client data, anonymise before sharing the folder.

## Labels OCR disagrees with

111 flagged. These are the first candidates for relabelling: the page was read confidently, but nothing on it resembles the labelled value.

- `client0000_writer06_p0000` **personal-details.isGoodHealth** - best partial match 40.0% at page confidence 0.95
- `client0000_writer06_p0000` **dependants[0].dateOfBirth** - best partial match 40.0% at page confidence 0.95
- `client0000_writer06_p0000` **dependants[0].relationshipType** - best partial match 50.0% at page confidence 0.95
- `client0000_writer06_p0000` **dependants[0].isForWholeOfLife** - best partial match 50.0% at page confidence 0.95
- `client0000_writer06_p0336` **dependants[0].dateOfBirth** - best partial match 35.29411764705882% at page confidence 0.97
- `client0001_writer08_p0001` **personal-details.drivingLicenceIssueDate** - best partial match 50.0% at page confidence 0.93
- `client0001_writer08_p0001` **personal-details.passportValidFrom** - best partial match 50.0% at page confidence 0.93
- `client0001_writer08_p0001` **personal-details.passportValidTo** - best partial match 40.0% at page confidence 0.93
- `client0001_writer08_p0001` **employment.occupation** - best partial match 47.36842105263158% at page confidence 0.93
- `client0001_writer08_p0337` **personal-details.drivingLicenceIssueDate** - best partial match 40.0% at page confidence 0.91
- `client0001_writer08_p0337` **personal-details.passportValidFrom** - best partial match 50.0% at page confidence 0.91
- `client0001_writer08_p0337` **personal-details.passportValidTo** - best partial match 40.0% at page confidence 0.91
- `client0001_writer08_p0337` **employment.bonusAmount** - best partial match 50.0% at page confidence 0.91
- `client0001_writer08_p0337` **employment.bonusFrequency** - best partial match 50.0% at page confidence 0.91
- `client0002_writer06_p0338` **life-style.heightBeforePregnancy** - best partial match 42.85714285714286% at page confidence 0.95
- `client0003_writer04_p0003` **personal-details.dateOfBirth** - best partial match 40.0% at page confidence 0.96
- `client0003_writer04_p0003` **personal-details.isGoodHealth** - best partial match 50.0% at page confidence 0.96
- `client0003_writer04_p0003` **personal-details.hasWill** - best partial match 50.0% at page confidence 0.96
- `client0003_writer04_p0003` **employment.startDate** - best partial match 19.999999999999996% at page confidence 0.96
- `client0003_writer04_p0003` **employment.bonusFrequency** - best partial match 50.0% at page confidence 0.96
- `client0003_writer04_p0339` **personal-details.dateOfBirth** - best partial match 40.0% at page confidence 0.94
- `client0003_writer04_p0339` **personal-details.isGoodHealth** - best partial match 50.0% at page confidence 0.94
- `client0003_writer04_p0339` **personal-details.hasWill** - best partial match 50.0% at page confidence 0.94
- `client0003_writer04_p0339` **employment.startDate** - best partial match 30.000000000000004% at page confidence 0.94
- `client0003_writer04_p0339` **employment.bonusFrequency** - best partial match 50.0% at page confidence 0.94

## Gaps - what to add

102 of 107 fields have fewer examples than needed to demonstrate their wrong-fill target.

Implied extra pages (rough): **1000**

- 39 near-duplicate pairs found: these inflate accuracy and risk leaking across splits.

| field | kind | have | need | add |
| --- | --- | ---: | ---: | ---: |
| `company-details.primaryPhone` | phone | 34 | 400 | **366** |
| `company-details.primaryEmail` | email | 41 | 400 | **359** |
| `company-details.telephoneNumber` | phone | 53 | 400 | **347** |
| `personal-details.niNumber` | niNumber | 78 | 400 | **322** |
| `credit.anyOtherLending` | money | 5 | 300 | **295** |
| `credit.loansHp` | money | 7 | 300 | **293** |
| `credit.mortgageMainResidence` | money | 11 | 300 | **289** |
| `credit.creditCards` | money | 12 | 300 | **288** |
| `credit.overdraft` | money | 12 | 300 | **288** |
| `credit.mortgageOtherProperty` | money | 13 | 300 | **287** |
| `life-style.lastSmokedMonths` | number | 14 | 300 | **286** |
| `life-style.lastSmokedYears` | number | 20 | 300 | **280** |
| `expenditure.insurances` | money | 25 | 300 | **275** |
| `life-style.weightBeforePregnancyKg` | number | 25 | 300 | **275** |
| `expenditure.anticipatesIncreaseNextFiveYears` | yesNo | 26 | 300 | **274** |
| `expenditure.entertainment` | money | 27 | 300 | **273** |
| `expenditure.investments` | money | 27 | 300 | **273** |
| `expenditure.miscellaneous` | money | 30 | 300 | **270** |
| `expenditure.pensionContributions` | money | 31 | 300 | **269** |
| `expenditure.holidays` | money | 32 | 300 | **268** |
| `expenditure.householdMaintenance` | money | 35 | 300 | **265** |
| `employment.overtimeFrequency` | choice | 38 | 300 | **262** |
| `expenditure.educationFees` | money | 39 | 300 | **261** |
| `expenditure.foodAndClothing` | money | 39 | 300 | **261** |
| `company-details.grossProfit` | money | 39 | 300 | **261** |
| `employment.commissionFrequency` | choice | 40 | 300 | **260** |
| `assets.allOtherAssets` | money | 40 | 300 | **260** |
| `company-details.netProfit` | money | 40 | 300 | **260** |
| `company-details.totalValueOfBusiness` | money | 40 | 300 | **260** |
| `employment.bonusFrequency` | choice | 42 | 300 | **258** |
| `life-style.doesHazardousSports` | yesNo | 43 | 300 | **257** |
| `employment.overtimeAmount` | money | 44 | 300 | **256** |
| `expenditure.otherMortgageOrRent` | money | 44 | 300 | **256** |
| `expenditure.communications` | money | 44 | 300 | **256** |
| `expenditure.transportCosts` | money | 44 | 300 | **256** |
| `assets.cashDeposits` | money | 44 | 300 | **256** |
| `company-details.numberOfEmployees` | number | 44 | 300 | **256** |
| `employment.commissionAmount` | money | 45 | 300 | **255** |
| `expenditure.councilTax` | money | 45 | 300 | **255** |
| `expenditure.studentLoan` | money | 45 | 300 | **255** |
| `assets.businessAssets` | money | 45 | 300 | **255** |
| `assets.property` | money | 46 | 300 | **254** |
| `assets.investmentBonds` | money | 46 | 300 | **254** |
| `employment.grossAnnualSalary` | money | 47 | 300 | **253** |
| `expenditure.utilityBills` | money | 47 | 300 | **253** |
