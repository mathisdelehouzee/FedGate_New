# MixedDB dataset overview

This file explains the content of each folder and CSV inside MixedDB.

## Root folders
- ADNI-skull/: skull-stripped T1 MRI NIfTI files (.nii.gz) organized by ADNI subject id.
- OASIS-skull/: skull-stripped T1 MRI NIfTI files (.nii.gz) organized by OASIS subject id.
- NACC-skull/: skull-stripped T1 MRI NIfTI files (.nii.gz) organized by NACC subject id.
- data/: CSV tables and train/val/test splits used for tabular and MRI experiments.

## data/adni (curated ADNI tables)
- ALL_4class_clinical.csv: clinical and cognitive features with 4-class labels. Key fields: PTID, VISCODE, DIAGNOSIS, Group, CLASS_4, BL_DX, LAST_DX, DX, AGE, BMI, nii_path.
- adni_cn_ad.csv: CN vs AD subset with MRI path and clinical features (subject_id, scan_path, DX, AGE, PTGENDER, MMSCORE, CDGLOBAL, BMI).
- adni_cn_ad_trajectory.csv: same schema as adni_cn_ad.csv, typically used for trajectory analysis (has trajectory field).
- clinical_data_all_groups.csv: clinical features for all groups (CN/MCI/AD) with Group and nii_path.
- clinical_tabular_data.csv: compact tabular set with entry info (entry_age, entry_visit, entry_date) and mri_count.
- dxsum.csv: diagnostic summary per visit (DX flags, confidence, etc).
- mci_stable_patients.csv: subset of stable MCI patients, same schema as clinical_data_all_groups.csv.
- mri_cn_ad_train.csv, mri_cn_ad_val.csv, mri_cn_ad_test.csv: MRI splits for CN vs AD. Columns: scan_path, subject_id, group, label, source.

### data/adni/tabular_raw (raw ADNI exports)
- 3D_MPRAGE_Imaging_Cohort_ADVERSE_10Oct2025.csv: adverse events.
- 3D_MPRAGE_Imaging_Cohort_ANTIAMYTX_10Oct2025.csv: anti-amyloid treatments.
- 3D_MPRAGE_Imaging_Cohort_BACKMEDS_10Oct2025.csv: baseline medications.
- 3D_MPRAGE_Imaging_Cohort_BLSCHECK_10Oct2025.csv: baseline symptom checklist.
- 3D_MPRAGE_Imaging_Cohort_DXSUM_10Oct2025.csv: diagnosis summary.
- 3D_MPRAGE_Imaging_Cohort_INITHEALTH_10Oct2025.csv: initial health history.
- 3D_MPRAGE_Imaging_Cohort_Key_MRI_10Oct2025.csv: MRI acquisition metadata (image_id, series_description, scanner info).
- 3D_MPRAGE_Imaging_Cohort_MEDHIST_10Oct2025.csv: medical history and risk factors.
- 3D_MPRAGE_Imaging_Cohort_My_Table_10Oct2025.csv: custom merged table of demographics, tests, vitals.
- 3D_MPRAGE_Imaging_Cohort_NEUROBAT_10Oct2025.csv: neuropsych battery (memory, language, executive tests).
- 3D_MPRAGE_Imaging_Cohort_NEUROEXM_10Oct2025.csv: neurologic exam.
- 3D_MPRAGE_Imaging_Cohort_PHYSICAL_10Oct2025.csv: physical exam.
- 3D_MPRAGE_Imaging_Cohort_RECADV_10Oct2025.csv: recent adverse events.
- 3D_MPRAGE_Imaging_Cohort_RECBLLOG_10Oct2025.csv: baseline symptom log.
- 3D_MPRAGE_Imaging_Cohort_RECCMEDS_10Oct2025.csv: current medications.
- 3D_MPRAGE_Imaging_Cohort_RECMHIST_10Oct2025.csv: recent medical history.
- 3D_MPRAGE_Imaging_Cohort_Study_Entry_10Oct2025.csv: study entry info (entry_age, entry_visit, entry_date).
- 3D_MPRAGE_Imaging_Cohort_VITALS_10Oct2025.csv: vitals (weight, height, BP, etc).

## data/oasis (OASIS3 UDS tables and derived sets)
- OASIS3_UDSa1_participant_demo.csv: participant demographics and living situation.
- OASIS3_UDSa2_cs_demo.csv: co-participant/informant demographics.
- OASIS3_UDSa3.csv: family history and relatives (large UDS A3 form).
- OASIS3_UDSa4D_med_codes.csv: medications as coded drug ids.
- OASIS3_UDSa4G_med_names.csv: medications as drug names.
- OASIS3_UDSa5_health_history.csv: medical history and risk factors.
- OASIS3_UDSb1_physical_eval.csv: physical evaluation (weight, height, BP, vision, hearing).
- OASIS3_UDSb2_his_cvd.csv: cerebrovascular history and related flags.
- OASIS3_UDSb3.csv: motor/neurologic exam items (tremor, rigidity, gait, etc).
- OASIS3_UDSb4_cdr.csv: CDR and MMSE summary with clinician diagnosis codes.
- OASIS3_UDSb5_npiq.csv: Neuropsychiatric Inventory Questionnaire.
- OASIS3_UDSb6_gds.csv: Geriatric Depression Scale.
- OASIS3_UDSb7_faq_fas.csv: Functional Activities Questionnaire (FAQ/FAS).
- OASIS3_UDSb8_neuro_exam.csv: detailed neurologic exam flags.
- OASIS3_UDSb9_symptoms.csv: cognitive/behavior/motor symptom timeline.
- OASIS3_UDSc1_cognitive_assessments.csv: cognitive test battery and MoCA.
- OASIS3_UDSd1_diagnoses.csv: diagnostic flags (CN/MCI/AD and other etiologies).
- OASIS3_UDSd2_med_conditions.csv: medical conditions (comorbidities).
- oasis_all.csv: merged tabular features with derived DX and clinical scores.
- oasis_all_full.csv: same schema as oasis_all.csv.
- oasis_tabular.csv: curated tabular dataset for modeling (same schema as oasis_all.csv).
- oasis_mri.csv: same schema as oasis_tabular.csv; name indicates MRI-matched subset.
- oasis-t1_12_16_2025.csv: T1 image list/export (Image Data ID, Subject, Modality, Format, Downloaded).
- mri_cn_ad_train.csv, mri_cn_ad_val.csv, mri_cn_ad_test.csv: MRI splits for CN vs AD with scan_path/label fields.

## data/nacc (NACC derived tables and exports)
- idaSearch_12_16_2025.csv: IDA search export (Subject ID, Sex, Age, Description).
- investigator_fcsf_nacc71.csv: CSF biomarkers (ABETA, PTAU, TTAU) with collection dates.
- investigator_ftldlbd_nacc71.csv: full NACC investigator dataset (very wide; demographics, history, neuropsych, diagnosis, imaging, neuropathology).
- investigator_mri_nacc71.csv: MRI metadata and volumetric measures.
- investigator_scan_mri_nacc71.zip: scan list export for MRI.
- investigator_scan_pet_nacc71.zip: scan list export for PET.
- nacc_tabular.csv: curated CN vs AD tabular features (Subject, DX, AGE, PTGENDER, PTEDUCAT, PTMARRY, tests, vitals).
- nacc_tabular_mri.csv: same schema as nacc_tabular.csv; name indicates MRI-matched subset.
- nacc_tabular_t1.csv: same schema as nacc_tabular.csv; name indicates T1-matched subset.
- nacc-t1_12_16_2025.csv: T1 image list/export (Image Data ID, Subject, Group, Modality, Format, Downloaded).

## data/mri (ADNI split lists)
- adni_cn_ad/: all.csv, train.csv, val.csv, test.csv with scan_path, subject_id, group, label, source; config.yaml holds split settings.
- adni_cn_ad_trajectory/: all.csv, train.csv, val.csv, test.csv with subject_id, scan_path, DX, RID, source, group, label; config.yaml and metadata.json store split metadata.

## data/combined (cross-dataset splits)
- mri_cn_ad_train.csv, mri_cn_ad_val.csv, mri_cn_ad_test.csv: combined CN vs AD MRI splits across ADNI, OASIS, and NACC.

## Common column notes
- subject identifiers: PTID (ADNI), Subject (OASIS/NACC), subject_id (derived tables)
- visit identifiers: VISCODE, VISIT, OASIS_session_label
- diagnosis: DX, DIAGNOSIS, Group, label
- imaging paths: scan_path or nii_path (path to .nii.gz)
- demographics/tests: AGE, PTGENDER, PTEDUCAT, PTMARRY, MMSCORE, CDGLOBAL, CATANIMSC, TRAASCOR, TRABSCOR

See DATASETS.md for global stats and dataset definitions.
