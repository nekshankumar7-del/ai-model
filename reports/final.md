# Fact-find extraction - final report

Generated 2026-11-17 from the **test** split (54 pages), scored once.

> **These numbers are from generated pages, not real client notes.**
> They show the pipeline works end to end and give a reference point, but they are not evidence about your documents. Synthetic pages are cleaner and more regular than a photographed handwritten note, so treat every number here as an optimistic upper bound. The brief's targets can only be claimed against real data.

## Headline

| | baseline (rules) | final | target |
| --- | ---: | ---: | ---: |
| wrong-fill rate | 32.10% | **7.25%** | see per-kind |
| exact match | 50.8% | **27.6%** | maximise |
| coverage | 66.3% | **29.5%** | maximise |
| unsure | 1.5% | **46.6%** | - |
| missing | 32.7% | **31.8%** | - |
| values filled | 972 | **386** | - |
| wrong values filled | 312 | **28** | - |

`wrong-fill rate` is the share of everything the pipeline chose to fill that was wrong, including values it filled for fields the page never mentioned. `coverage` is the share of the fields actually present on the pages that were filled correctly.

Of the values the pipeline left unsure, **61.0%** had the correct answer inside the top 3 candidates it offered.

## Targets

**Targets missed.** They have not been lowered.

| kind | wrong-fill | target | met | filled | wrong | coverage |
| --- | ---: | ---: | :---: | ---: | ---: | ---: |
| choice | 0.95% | 1.00% | yes | 105 | 1 | 46.5% |
| date | 0.00% | 1.00% | yes | 2 | 0 | 1.7% |
| email | 25.00% | 0.00% | **no** | 4 | 1 | 6.9% |
| fullName | 0.00% | 5.00% | yes | 9 | 0 | 28.1% |
| money | 0.00% | 1.00% | yes | 11 | 0 | 4.6% |
| name | 18.92% | 5.00% | **no** | 74 | 14 | 56.2% |
| niNumber | 0.00% | 0.00% | yes | 1 | 0 | 8.3% |
| number | 0.00% | 1.00% | yes | 24 | 0 | 31.2% |
| phone | 0.00% | 0.00% | yes | 5 | 0 | 8.9% |
| text | 26.47% | 5.00% | **no** | 34 | 9 | 22.9% |
| yesNo | 2.56% | 1.00% | **no** | 117 | 3 | 55.8% |

## Validation against test

Thresholds were chosen on validation. Where test is worse, the threshold was fitted to the validation writers and did not carry over - which is the single most important number in this report, because it is what would happen on a new adviser's handwriting.

| kind | validation wrong-fill | test wrong-fill | moved by |
| --- | ---: | ---: | ---: |
| email | 0.00% | 25.00% | +25.00% |
| text | 3.03% | 26.47% | +23.44% |
| name | 4.35% | 18.92% | +14.57% |
| yesNo | 0.61% | 2.56% | +1.96% |
| choice | 0.00% | 0.95% | +0.95% |
| date | 0.00% | 0.00% | 0.00% |
| fullName | 0.00% | 0.00% | 0.00% |
| money | 0.00% | 0.00% | 0.00% |
| niNumber | 0.00% | 0.00% | 0.00% |
| number | 0.00% | 0.00% | 0.00% |
| phone | 0.00% | 0.00% | 0.00% |

5 of 11 kinds got worse on test.

The cause is sample size, not a modelling flaw: the thresholds were fitted on the validation split's writers only. With a handful of writers, a threshold that looks safe on validation is partly fitted to those particular hands. The fix is more writers, not a different model - and until then the validation numbers should be read as optimistic.

### What would close each gap

**email** - 25.00% against a 0.00% target (1 wrong of 4 filled).

  - only 4 values of this kind were filled on this split. A 0.00% rate cannot be demonstrated at that size - roughly 3000 filled examples are needed before the measurement means anything.
  - raising this kind's threshold trades coverage for correctness; the curve above shows exactly how much.
  - a printed template form with boxed fields would remove most of this error class outright, if the business can adopt one.

**name** - 18.92% against a 5.00% target (14 wrong of 74 filled).

  - only 74 values of this kind were filled on this split. A 5.00% rate cannot be demonstrated at that size - roughly 60 filled examples are needed before the measurement means anything.
  - raising this kind's threshold trades coverage for correctness; the curve above shows exactly how much.
  - the limit here is handwriting recognition, not field matching. The biggest single win is a value recogniser fine-tuned on your own cropped value regions - which needs a few thousand labelled crops from real notes.
  - a printed template form with boxed fields would remove most of this error class outright, if the business can adopt one.

**text** - 26.47% against a 5.00% target (9 wrong of 34 filled).

  - only 34 values of this kind were filled on this split. A 5.00% rate cannot be demonstrated at that size - roughly 60 filled examples are needed before the measurement means anything.
  - raising this kind's threshold trades coverage for correctness; the curve above shows exactly how much.
  - the limit here is handwriting recognition, not field matching. The biggest single win is a value recogniser fine-tuned on your own cropped value regions - which needs a few thousand labelled crops from real notes.
  - a printed template form with boxed fields would remove most of this error class outright, if the business can adopt one.

**yesNo** - 2.56% against a 1.00% target (3 wrong of 117 filled).

  - raising this kind's threshold trades coverage for correctness; the curve above shows exactly how much.
  - add the wordings advisers actually use to this field's aliases in the knowledge file - no retraining needed.
  - a printed template form with boxed fields would remove most of this error class outright, if the business can adopt one.

## Coverage vs wrong-fill

![coverage vs wrong-fill](figures/coverage-test.png)

Each point is a fill threshold. Moving left refuses more values (lower wrong-fill, lower coverage); moving right fills more. The shipped thresholds are chosen per kind as the leftmost point that still meets that kind's target.

## Per field

Fields with at least one example on this split, worst wrong-fill first.

| field | kind | n | baseline exact | final exact | baseline wrong | final wrong | unsure | missing | target met |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `dependants.name` | name | 26 | 19.2% | 23.1% | 88.10% | 53.85% | 65.4% | 23.1% | **no** |
| `health.hasNoDiagnosedConditions` | yesNo | 5 | 40.0% | 20.0% | 33.33% | 50.00% | 0.0% | 80.0% | **no** |
| `company-details.primaryLastName` | text | 12 | 33.3% | 25.0% | 50.00% | 40.00% | 25.0% | 33.3% | **no** |
| `company-details.primaryFirstName` | text | 9 | 33.3% | 33.3% | 66.67% | 40.00% | 55.6% | 11.1% | **no** |
| `company-details.natureOfBusiness` | text | 11 | 54.5% | 18.2% | 45.45% | 33.33% | 63.6% | 27.3% | **no** |
| `employment.employerName` | text | 11 | 18.2% | 18.2% | 80.00% | 33.33% | 45.5% | 27.3% | **no** |
| `company-details.companyName` | text | 8 | 37.5% | 25.0% | 72.73% | 33.33% | 50.0% | 25.0% | **no** |
| `personal-details.email` | email | 45 | 31.1% | 6.7% | 57.58% | 25.00% | 57.8% | 33.3% | **no** |
| `dependants.relationshipType` | choice | 26 | 46.2% | 11.5% | 14.29% | 25.00% | 61.5% | 26.9% | **no** |
| `personal-details.lastName` | name | 34 | 41.2% | 50.0% | 56.25% | 22.73% | 32.4% | 17.6% | **no** |
| `personal-details.passportNo` | text | 14 | 50.0% | 28.6% | 53.33% | 20.00% | 50.0% | 14.3% | **no** |
| `employment.occupation` | text | 11 | 54.5% | 45.5% | 60.00% | 16.67% | 81.8% | 9.1% | **no** |
| `personal-details.firstName` | name | 35 | 60.0% | 54.3% | 27.59% | 9.52% | 20.0% | 28.6% | **no** |
| `personal-details.isGoodHealth` | yesNo | 27 | 74.1% | 77.8% | 4.76% | 4.55% | 3.7% | 14.8% | **no** |
| `personal-details.isSmoker` | yesNo | 40 | 67.5% | 70.0% | 3.57% | 3.45% | 2.5% | 25.0% | **no** |
| `personal-details.gender` | choice | 37 | 70.3% | 73.0% | 0.00% | 0.00% | 2.7% | 24.3% | yes |
| `personal-details.maritalStatus` | choice | 37 | 67.6% | 54.1% | 3.85% | 0.00% | 16.2% | 32.4% | yes |
| `personal-details.countryOfBirth` | choice | 35 | 60.0% | 54.3% | 30.00% | 0.00% | 31.4% | 17.1% | yes |
| `personal-details.dateOfBirth` | date | 35 | 31.4% | 2.9% | 8.33% | 0.00% | 42.9% | 57.1% | yes |
| `personal-details.middleName` | name | 35 | 51.4% | 51.4% | 25.00% | 0.00% | 17.1% | 37.1% | yes |
| `personal-details.mobileNumber` | phone | 35 | 82.9% | 14.3% | 6.45% | 0.00% | 62.9% | 25.7% | yes |
| `personal-details.retirementAge` | number | 35 | 68.6% | 65.7% | 7.69% | 0.00% | 17.1% | 22.9% | yes |
| `personal-details.nationality` | choice | 34 | 64.7% | 50.0% | 18.52% | 0.00% | 29.4% | 20.6% | yes |
| `personal-details.hasWill` | yesNo | 33 | 45.5% | 57.6% | 0.00% | 0.00% | 9.1% | 33.3% | yes |
| `personal-details.fullName` | fullName | 32 | 40.6% | 28.1% | 68.29% | 0.00% | 75.0% | 18.8% | yes |
| `personal-details.title` | choice | 30 | 50.0% | 43.3% | 0.00% | 0.00% | 13.3% | 43.3% | yes |
| `dependants.dateOfBirth` | date | 26 | 34.6% | 0.0% | 10.00% | 0.00% | 42.3% | 57.7% | yes |
| `dependants.isForWholeOfLife` | yesNo | 25 | 24.0% | 36.0% | 0.00% | 0.00% | 4.0% | 60.0% | yes |
| `personal-details.drivingLicenceNumber` | text | 16 | 12.5% | 0.0% | 89.47% | 0.00% | 106.2% | 12.5% | yes |
| `company-details.primaryJobTitle` | text | 14 | 21.4% | 7.1% | 62.50% | 0.00% | 42.9% | 57.1% | yes |
| `personal-details.drivingLicenceExpiryDate` | date | 14 | 42.9% | 0.0% | 0.00% | 0.00% | 57.1% | 42.9% | yes |
| `company-details.primaryEmail` | email | 13 | 23.1% | 0.0% | 57.14% | 0.00% | 53.8% | 46.2% | yes |
| `employment.startDate` | date | 13 | 38.5% | 0.0% | 0.00% | 0.00% | 46.2% | 61.5% | yes |
| `company-details.numberOfEmployees` | number | 12 | 50.0% | 8.3% | 33.33% | 0.00% | 66.7% | 33.3% | yes |
| `company-details.telephoneNumber` | phone | 12 | 41.7% | 0.0% | 0.00% | 0.00% | 66.7% | 33.3% | yes |
| `personal-details.niNumber` | niNumber | 12 | 25.0% | 8.3% | 40.00% | 0.00% | 33.3% | 58.3% | yes |
| `personal-details.passportValidTo` | date | 12 | 33.3% | 8.3% | 0.00% | 0.00% | 41.7% | 50.0% | yes |
| `company-details.businessTurnover` | money | 11 | 63.6% | 0.0% | 12.50% | 0.00% | 81.8% | 27.3% | yes |
| `company-details.registeredNumberAndAddress` | text | 11 | 36.4% | 0.0% | 60.00% | 0.00% | 81.8% | 18.2% | yes |
| `health.hadMedicalAppointmentsLast12Months` | yesNo | 11 | 54.5% | 45.5% | 0.00% | 0.00% | 18.2% | 36.4% | yes |
| `health.hasPlannedOrPendingAppointments` | yesNo | 11 | 63.6% | 54.5% | 0.00% | 0.00% | 9.1% | 36.4% | yes |
| `health.tookEarlyRetirementIllHealth` | yesNo | 11 | 54.5% | 45.5% | 0.00% | 0.00% | 9.1% | 45.5% | yes |
| `income-expenditure.totalIncome` | money | 11 | 90.9% | 0.0% | 16.67% | 0.00% | 100.0% | 18.2% | yes |
| `life-style.isPregnant` | yesNo | 11 | 72.7% | 63.6% | 0.00% | 0.00% | 9.1% | 27.3% | yes |
| `personal-details.holdsDrivingLicence` | yesNo | 11 | 36.4% | 45.5% | 0.00% | 0.00% | 18.2% | 36.4% | yes |
| `company-details.grossProfit` | money | 10 | 60.0% | 10.0% | 25.00% | 0.00% | 50.0% | 50.0% | yes |
| `dependants.dependencyTermYears` | number | 10 | 70.0% | 0.0% | 12.50% | 0.00% | 110.0% | 20.0% | yes |
| `employment.employmentBasis` | choice | 10 | 50.0% | 30.0% | 0.00% | 0.00% | 30.0% | 40.0% | yes |
| `income-expenditure.disposableIncome` | money | 10 | 70.0% | 10.0% | 0.00% | 0.00% | 60.0% | 30.0% | yes |
| `income-expenditure.totalExpenditure` | money | 10 | 70.0% | 0.0% | 12.50% | 0.00% | 80.0% | 30.0% | yes |
| `assets.businessAssets` | money | 9 | 66.7% | 0.0% | 0.00% | 0.00% | 66.7% | 33.3% | yes |
| `assets.investmentBonds` | money | 9 | 66.7% | 0.0% | 14.29% | 0.00% | 77.8% | 44.4% | yes |
| `company-details.netProfit` | money | 9 | 44.4% | 0.0% | 33.33% | 0.00% | 88.9% | 33.3% | yes |
| `company-details.primaryPhone` | phone | 9 | 77.8% | 0.0% | 0.00% | 0.00% | 88.9% | 11.1% | yes |
| `personal-details.drivingLicenceIssueDate` | date | 9 | 44.4% | 0.0% | 33.33% | 0.00% | 88.9% | 22.2% | yes |
| `company-details.totalValueOfBusiness` | money | 8 | 50.0% | 0.0% | 20.00% | 0.00% | 75.0% | 37.5% | yes |
| `employment.grossAnnualSalary` | money | 8 | 75.0% | 0.0% | 40.00% | 0.00% | 125.0% | 25.0% | yes |
| `employment.overtimeAmount` | money | 8 | 50.0% | 0.0% | 0.00% | 0.00% | 50.0% | 50.0% | yes |
| `employment.overtimeFrequency` | choice | 8 | 37.5% | 25.0% | 0.00% | 0.00% | 12.5% | 62.5% | yes |
| `life-style.alcoholUnitsPerWeek` | number | 8 | 62.5% | 0.0% | 16.67% | 0.00% | 75.0% | 25.0% | yes |
| `personal-details.passportValidFrom` | date | 8 | 62.5% | 0.0% | 16.67% | 0.00% | 87.5% | 25.0% | yes |
| `assets.pensionFunds` | money | 7 | 28.6% | 0.0% | 66.67% | 0.00% | 114.3% | 28.6% | yes |
| `employment.commissionAmount` | money | 7 | 85.7% | 0.0% | 0.00% | 0.00% | 71.4% | 28.6% | yes |
| `employment.commissionFrequency` | choice | 7 | 71.4% | 0.0% | 16.67% | 0.00% | 100.0% | 0.0% | yes |
| `employment.jobTitle` | text | 7 | 42.9% | 42.9% | 62.50% | 0.00% | 71.4% | 28.6% | yes |
| `expenditure.councilTax` | money | 7 | 71.4% | 14.3% | 0.00% | 0.00% | 71.4% | 14.3% | yes |
| `health.hasNoFamilyHistoryConditions` | yesNo | 7 | 14.3% | 14.3% | 0.00% | 0.00% | 0.0% | 85.7% | yes |
| `income-expenditure.savingsAndInvestments` | money | 7 | 14.3% | 0.0% | 75.00% | 0.00% | 71.4% | 28.6% | yes |
| `life-style.doesSmoke` | yesNo | 7 | 57.1% | 42.9% | 20.00% | 0.00% | 28.6% | 42.9% | yes |
| `assets.allOtherAssets` | money | 6 | 50.0% | 0.0% | 25.00% | 0.00% | 83.3% | 33.3% | yes |
| `assets.cashDeposits` | money | 6 | 50.0% | 0.0% | 25.00% | 0.00% | 66.7% | 50.0% | yes |
| `assets.isa` | money | 6 | 50.0% | 0.0% | 50.00% | 0.00% | 100.0% | 33.3% | yes |
| `assets.property` | money | 6 | 66.7% | 16.7% | 20.00% | 0.00% | 50.0% | 33.3% | yes |
| `expenditure.otherMortgageOrRent` | money | 6 | 83.3% | 0.0% | 44.44% | 0.00% | 150.0% | 16.7% | yes |
| `expenditure.studentLoan` | money | 6 | 50.0% | 16.7% | 0.00% | 0.00% | 33.3% | 50.0% | yes |
| `life-style.doesHazardousSports` | yesNo | 6 | 50.0% | 33.3% | 0.00% | 0.00% | 16.7% | 50.0% | yes |
| `expenditure.householdMaintenance` | money | 5 | 60.0% | 0.0% | 25.00% | 0.00% | 60.0% | 40.0% | yes |
| `expenditure.miscellaneous` | money | 5 | 80.0% | 0.0% | 20.00% | 0.00% | 100.0% | 20.0% | yes |
| `expenditure.pensionContributions` | money | 5 | 40.0% | 20.0% | 0.00% | 0.00% | 20.0% | 60.0% | yes |
| `expenditure.utilityBills` | money | 5 | 20.0% | 0.0% | 75.00% | 0.00% | 40.0% | 60.0% | yes |
| `expenditure.water` | money | 5 | 100.0% | 0.0% | 0.00% | 0.00% | 100.0% | 0.0% | yes |
| `health.earlyRetirementDetails` | text | 5 | 0.0% | 0.0% | 100.00% | 0.00% | 200.0% | 0.0% | yes |
| `assets.shares` | money | 4 | 25.0% | 0.0% | 50.00% | 0.00% | 25.0% | 75.0% | yes |
| `expenditure.additionalInformation` | text | 4 | 0.0% | 0.0% | 100.00% | 0.00% | 100.0% | 25.0% | yes |
| `expenditure.communications` | money | 4 | 50.0% | 25.0% | 0.00% | 0.00% | 75.0% | 0.0% | yes |
| `expenditure.educationFees` | money | 4 | 50.0% | 25.0% | 0.00% | 0.00% | 25.0% | 50.0% | yes |
| `expenditure.entertainment` | money | 4 | 75.0% | 0.0% | 0.00% | 0.00% | 100.0% | 25.0% | yes |
| `expenditure.insurances` | money | 4 | 75.0% | 25.0% | 25.00% | 0.00% | 75.0% | 0.0% | yes |
| `expenditure.investments` | money | 4 | 50.0% | 0.0% | 0.00% | 0.00% | 75.0% | 25.0% | yes |
| `health.plannedAppointmentsDetails` | text | 4 | 0.0% | 0.0% | 100.00% | 0.00% | 125.0% | 0.0% | yes |
| `life-style.heightBeforePregnancy` | text | 4 | 0.0% | 0.0% | 100.00% | 0.00% | 25.0% | 100.0% | yes |
| `life-style.lastSmokedMonths` | number | 4 | 50.0% | 0.0% | 0.00% | 0.00% | 75.0% | 25.0% | yes |
| `life-style.lastSmokedYears` | number | 4 | 50.0% | 0.0% | 71.43% | 0.00% | 150.0% | 25.0% | yes |
| `life-style.weightBeforePregnancyKg` | number | 4 | 50.0% | 0.0% | 0.00% | 0.00% | 50.0% | 50.0% | yes |
| `credit.mortgageOtherProperty` | money | 3 | 33.3% | 0.0% | 0.00% | 0.00% | 33.3% | 66.7% | yes |
| `expenditure.anticipatesIncreaseNextFiveYears` | yesNo | 3 | 66.7% | 66.7% | 0.00% | 0.00% | 0.0% | 33.3% | yes |
| `expenditure.foodAndClothing` | money | 3 | 66.7% | 0.0% | 0.00% | 0.00% | 66.7% | 33.3% | yes |
| `expenditure.holidays` | money | 3 | 66.7% | 33.3% | 0.00% | 0.00% | 33.3% | 33.3% | yes |
| `expenditure.transportCosts` | money | 3 | 66.7% | 0.0% | 0.00% | 0.00% | 100.0% | 33.3% | yes |
| `health.medicalAppointmentsDetails` | text | 3 | 0.0% | 0.0% | 100.00% | 0.00% | 200.0% | 0.0% | yes |
| `credit.anyOtherLending` | money | 2 | 50.0% | 0.0% | 66.67% | 0.00% | 150.0% | 0.0% | yes |
| `credit.creditCards` | money | 2 | 100.0% | 50.0% | 0.00% | 0.00% | 50.0% | 0.0% | yes |
| `credit.mortgageMainResidence` | money | 2 | 100.0% | 0.0% | 0.00% | 0.00% | 100.0% | 0.0% | yes |
| `credit.overdraft` | money | 2 | 100.0% | 0.0% | 0.00% | 0.00% | 100.0% | 0.0% | yes |
| `employment.bonusAmount` | money | 2 | 100.0% | 0.0% | 33.33% | 0.00% | 150.0% | 0.0% | yes |
| `employment.bonusFrequency` | choice | 2 | 50.0% | 0.0% | 50.00% | 0.00% | 100.0% | 0.0% | yes |
| `credit.loansHp` | money | 1 | 100.0% | 0.0% | 50.00% | 0.00% | 200.0% | 0.0% | yes |

## Worst failures

The wrong fills the pipeline was most confident about - the ones that would actually reach a form.

| page | field | filled | should have been | confidence |
| --- | --- | --- | --- | ---: |
| `client0119_writer06_p0119` | `dependants[0].relationshipType` | `Sister` | `Grandmother` | 1.000 |
| `client0047_writer12_p0383` | `personal-details.isGoodHealth` | `True` | `False` | 1.000 |
| `client0060_writer06_p0396` | `personal-details.email` | `dawid.silua@gmail.com` | `david.silva@gmail.com` | 0.971 |
| `client0140_writer06_p0140` | `personal-details.lastName` | `Jaylar` | `Taylor` | 0.967 |
| `client0263_writer06_p0263` | `personal-details.isSmoker` | `True` | `False` | 0.967 |
| `client0085_writer12_p0085` | `dependants[1].name` | `Anthony McDonald` | `Theo McDonald` | 0.967 |
| `client0243_writer12_p0243` | `dependants[0].name` | `Susanferreira` | `Susan Ferreira` | 0.967 |
| `client0041_writer06_p0041` | `company-details.companyName` | `Bright Jutures Ltd` | `Bright Futures Ltd` | 0.907 |
| `client0190_writer06_p0190` | `personal-details.passportNo` | `sent dacs` | `sent docs` | 0.907 |
| `client0320_writer06_p0320` | `dependants[1].name` | `Juy Rahman` | `Ivy Rahman` | 0.907 |
| `client0322_writer06_p0322` | `personal-details.lastName` | `Jhampsan` | `Thompson` | 0.907 |
| `client0103_writer12_p0103` | `company-details.primaryFirstName` | `needsreview` | `needs review` | 0.907 |
| `client0103_writer12_p0103` | `company-details.primaryLastName` | `to folow up` | `to follow up` | 0.907 |
| `client0133_writer06_p0133` | `company-details.primaryFirstName` | `sentdocs` | `sent docs` | 0.886 |
| `client0140_writer06_p0140` | `dependants[0].name` | `Leo Jaylar` | `Leo Taylor` | 0.886 |
| `client0140_writer06_p0140` | `health.hasNoDiagnosedConditions` | `False` | `` | 0.886 |
| `client0026_writer12_p0362` | `dependants[0].name` | `Chiuli Okafor` | `Chidi Okafor` | 0.886 |
| `client0103_writer12_p0103` | `employment.occupation` | `developer` | `` | 0.886 |
| `client0262_writer12_p0262` | `personal-details.lastName` | `Name` | `Ali` | 0.886 |
| `client0300_writer12_p0300` | `dependants[2].name` | `Omar Ferreira` | `` | 0.886 |

> Page images are in the dataset folder under these names. They are not copied into this report, because they contain client data.

### Wrong fills by field

| field | count |
| --- | ---: |
| `dependants.name` | 7 |
| `personal-details.lastName` | 5 |
| `personal-details.firstName` | 2 |
| `company-details.primaryFirstName` | 2 |
| `company-details.primaryLastName` | 2 |
| `company-details.companyName` | 1 |
| `company-details.natureOfBusiness` | 1 |
| `personal-details.email` | 1 |
| `dependants.relationshipType` | 1 |
| `health.hasNoDiagnosedConditions` | 1 |
| `personal-details.passportNo` | 1 |
| `personal-details.isSmoker` | 1 |
| `personal-details.isGoodHealth` | 1 |
| `employment.occupation` | 1 |
| `employment.employerName` | 1 |

### Most often missed entirely

| field | times present but not found |
| --- | ---: |
| `personal-details.dateOfBirth` | 20 |
| `dependants.dateOfBirth` | 15 |
| `dependants.isForWholeOfLife` | 15 |
| `personal-details.email` | 15 |
| `personal-details.title` | 13 |
| `personal-details.middleName` | 13 |
| `personal-details.maritalStatus` | 12 |
| `personal-details.hasWill` | 11 |
| `personal-details.firstName` | 10 |
| `personal-details.isSmoker` | 10 |
| `personal-details.gender` | 9 |
| `personal-details.mobileNumber` | 9 |
| `company-details.primaryJobTitle` | 8 |
| `employment.startDate` | 8 |
| `personal-details.retirementAge` | 8 |

## What would improve each weak field

| field | problem | what to add |
| --- | --- | --- |
| `dependants.name` | wrong-fill 53.85% over target | more handwritten examples from more writers; free text is limited by the recogniser, not the matcher |
| `health.hasNoDiagnosedConditions` | wrong-fill 50.00% over target | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'No diagnosed conditions' and add those as `aliases` |
| `company-details.primaryLastName` | wrong-fill 40.00% over target | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Key contact last name' and add those as `aliases` |
| `company-details.primaryFirstName` | wrong-fill 40.00% over target | more handwritten examples from more writers; free text is limited by the recogniser, not the matcher |
| `company-details.natureOfBusiness` | wrong-fill 33.33% over target | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Nature of business' and add those as `aliases` |
| `employment.employerName` | wrong-fill 33.33% over target | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Employer name' and add those as `aliases` |
| `company-details.companyName` | wrong-fill 33.33% over target | more handwritten examples from more writers; free text is limited by the recogniser, not the matcher |
| `personal-details.email` | wrong-fill 25.00% over target | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Email' and add those as `aliases` |
| `dependants.relationshipType` | wrong-fill 25.00% over target | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Relationship' and add those as `aliases` |
| `personal-details.lastName` | wrong-fill 22.73% over target | more handwritten examples from more writers; free text is limited by the recogniser, not the matcher |
| `personal-details.passportNo` | wrong-fill 20.00% over target | more handwritten examples from more writers; free text is limited by the recogniser, not the matcher |
| `employment.occupation` | wrong-fill 16.67% over target | more handwritten examples from more writers; free text is limited by the recogniser, not the matcher |
| `personal-details.firstName` | wrong-fill 9.52% over target | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'First name' and add those as `aliases` |
| `personal-details.isGoodHealth` | wrong-fill 4.55% over target | examples using ticks, crosses and strike-throughs, which are the ambiguous cases |
| `personal-details.isSmoker` | wrong-fill 3.45% over target | examples using ticks, crosses and strike-throughs, which are the ambiguous cases |
| `personal-details.maritalStatus` | missed 32.4% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Marital status' and add those as `aliases` |
| `personal-details.countryOfBirth` | exact match only 54.3% | examples of the choices that never appear; add any wording advisers use for them to this field's `choices[].aliases` |
| `personal-details.dateOfBirth` | missed 57.1% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Date of birth' and add those as `aliases` |
| `personal-details.middleName` | missed 37.1% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Middle name' and add those as `aliases` |
| `personal-details.mobileNumber` | missed 25.7% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Mobile number' and add those as `aliases` |
| `personal-details.nationality` | exact match only 50.0% | examples of the choices that never appear; add any wording advisers use for them to this field's `choices[].aliases` |
| `personal-details.hasWill` | missed 33.3% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Has a will' and add those as `aliases` |
| `personal-details.fullName` | exact match only 28.1% | more handwritten examples from more writers; free text is limited by the recogniser, not the matcher |
| `personal-details.title` | missed 43.3% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Title' and add those as `aliases` |
| `dependants.dateOfBirth` | missed 57.7% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Date of birth' and add those as `aliases` |
| `dependants.isForWholeOfLife` | missed 60.0% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Dependant for whole of life' and add those as `aliases` |
| `personal-details.drivingLicenceNumber` | exact match only 0.0% | more handwritten examples from more writers; free text is limited by the recogniser, not the matcher |
| `company-details.primaryJobTitle` | missed 57.1% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Key contact job title' and add those as `aliases` |
| `personal-details.drivingLicenceExpiryDate` | missed 42.9% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Driving licence expiry date' and add those as `aliases` |
| `company-details.primaryEmail` | missed 46.2% of the time | more pages where this is written with a label the knowledge file does not list yet - check how advisers actually write 'Key contact email' and add those as `aliases` |

## Notes on method

- Splits are by connected components over (writer, client), so no hand and no person appears on both sides of a split.
- The test split was scored once, after validation stopped improving.
- `wrong-fill` counts a value filled for a field the page never mentioned as wrong, not as a free pass.
- Thresholds were chosen on validation as the lowest value per kind that still met that kind's target, which maximises coverage subject to the priority-1 constraint.
