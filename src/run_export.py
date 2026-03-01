from allen_exporter.exporter import multi_session_export

cache, ids = multi_session_export(1, val_rate=0.2, test_rate=0.2, subsample_frac=1)