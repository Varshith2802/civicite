### Evaluation (extractive mode)

| metric | value |
|---|---|
| n | 34 |
| answerable | 26 |
| unanswerable | 8 |
| retrieval_recall@5 | 1.0 |
| retrieval_mrr | 1.0 |
| answer_accuracy | 0.846 |
| abstain_precision | 1.0 |
| abstain_recall | 0.75 |
| abstain_f1 | 0.857 |
| faithfulness | 1.0 |
| citation_precision | 0.886 |
| tool_date_accuracy | 1.0 |
| latency_ms_p50 | 0.8 |
| latency_ms_p95 | 1.1 |

| id | answerable | abstained | correct | rank | verification | answer |
|---|---|---|---|---|---|---|
| q01 | True | False | True | 1 | verified | You must report the move to Skatteverket within one week after you have moved [1]. The easiest way is the e-se... |
| q02 | True | False | True | 1 | verified | Forwarding of letters and parcels (eftersändning) is a separate paid service that is not handled by Skatteverk... |
| q03 | True | False | True | 1 | verified | You must report the move to Skatteverket within one week after you have moved [1]. Counting from Tuesday 10 Ma... |
| q04 | True | False | True | 1 | verified | The last digit is a check digit calculated with the Luhn algorithm [1]. |
| q05 | True | False | True | 1 | verified | A coordination number (samordningsnummer) is an identity number for people who are not, or have not been, regi... |
| q06 | True | False | True | 1 | verified | It has the same structure as a personal identity number, but 60 is added to the day of birth, so a person born... |
| q07 | True | False | True | 1 | verified | The income tax return must be submitted no later than 2 May [1]. If 2 May falls on a Saturday, Sunday or publi... |
| q08 | True | False | True | 1 | verified | The income tax return must be submitted no later than 2 May [1]. In 2026, Saturday 2 May 2026 is not a working... |
| q09 | True | False | True | 1 | verified | Individuals who had taxable income in Sweden during the previous calendar year normally receive an income tax ... |
| q10 | True | False | False | 1 | verified | Skatteverket issues identity cards to people who are registered in the Swedish population register (folkbokför... |
| q11 | True | False | True | 1 | verified | If you do not have one, a person who knows you and holds a valid Swedish identity document can vouch for your ... |
| q12 | True | False | True | 1 | verified | Parents can receive parental benefit for a total of 480 days per child [1]. When the parents have joint custod... |
| q13 | True | False | True | 1 | verified | Parents can receive parental benefit for a total of 480 days per child [1]. When the parents have joint custod... |
| q14 | True | False | True | 1 | verified | The employer pays sick pay (sjuklön) for the first 14 days of the sick period [1]. |
| q15 | True | False | True | 1 | verified | From the eighth day of the sick period you need a medical certificate (läkarintyg) that shows how the illness ... |
| q16 | True | False | True | 1 | verified | Temporary parental benefit can be paid for up to 120 days per child and year [1]. Parental benefit can be used... |
| q17 | True | False | True | 1 | verified | You must apply for the benefit no later than 90 days after the first day you stayed home [1]. Counting from Th... |
| q18 | True | False | True | 1 | verified | Student aid (studiemedel) from CSN consists of a grant (bidrag), which you do not pay back, and a loan (lån), ... |
| q19 | True | False | False | 1 | verified | To keep receiving student aid, you need to pass your studies at a sufficient pace; CSN checks your study resul... |
| q20 | True | False | True | 1 | verified | If you become unemployed, register as a job seeker with Arbetsförmedlingen on your first day of unemployment [... |
| q21 | True | False | True | 1 | verified | Unemployment benefit is paid by an unemployment insurance fund (a-kassa), not by Arbetsförmedlingen [1]. You a... |
| q22 | True | False | True | 1 | verified | Swedish for immigrants (svenska för invandrare, SFI) is a free basic Swedish language course for adults whose ... |
| q23 | True | False | False | 1 | verified | A valid licence from a country outside the EU/EEA can be used in Sweden for a limited time [1]. A valid drivin... |
| q24 | True | False | True | 1 | verified | The most widely used e-identification in Sweden is BankID, which is issued by banks [1]. |
| q25 | True | False | False | 1 | verified | If 2 May falls on a Saturday, Sunday or public holiday, the deadline moves to the next working day [1]. If you... |
| q26 | True | False | True | 1 | verified | Parental benefit can be used until the child turns 12 or finishes the fifth year of school [1]. After the chil... |
| u01 | False | True | False | None | no-claims | I could not find this in the documents I have. Please contact the relevant agency directly. |
| u02 | False | True | False | None | no-claims | I could not find this in the documents I have. Please contact the relevant agency directly. |
| u03 | False | True | False | None | no-claims | I could not find this in the documents I have. Please contact the relevant agency directly. |
| u04 | False | True | False | None | no-claims | I could not find this in the documents I have. Please contact the relevant agency directly. |
| u05 | False | False | False | None | verified | You can receive temporary parental benefit when you stay home from work to care for a sick child who is under ... |
| u06 | False | False | False | None | verified | Student aid (studiemedel) from CSN consists of a grant (bidrag), which you do not pay back, and a loan (lån), ... |
| u07 | False | True | False | None | no-claims | I could not find this in the documents I have. Please contact the relevant agency directly. |
| u08 | False | True | False | None | no-claims | Jag kunde inte hitta detta i de dokument jag har. Please contact the relevant agency directly. |

### Abstention threshold sweep

| threshold | abstain_f1 | answer_accuracy |
|---|---|---|
| 0.1 | 0.4 | 0.846 |
| 0.2 | 0.667 | 0.846 |
| 0.25 | 0.769 | 0.846 |
| 0.3 | 0.857 | 0.846 |
| 0.34 | 0.857 | 0.846 |
| 0.4 | 0.875 | 0.808 |
| 0.5 | 0.842 | 0.769 |
| 0.6 | 0.727 | 0.654 |
