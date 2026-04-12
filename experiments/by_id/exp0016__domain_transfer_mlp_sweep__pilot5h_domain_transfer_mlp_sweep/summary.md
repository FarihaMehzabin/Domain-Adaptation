# Domain Transfer MLP Sweep Summary

| Rank | Backbone | Hidden | D0 Val AUROC | D0 Test AUROC | D1 AUROC | D2 AUROC | Delta vs Linear D0 | Delta vs Linear D1 | Delta vs Linear D2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 | 1024 | 0.852728 | 0.847347 | 0.836898 | 0.502939 | 0.001849 | -0.008534 | 0.002233 |
| 2 | exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 | 512 | 0.851352 | 0.843756 | 0.842082 | 0.505566 | -0.001741 | -0.003349 | 0.004860 |
| 3 | exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 | 256 | 0.851339 | 0.846845 | 0.820078 | 0.500139 | 0.001347 | -0.025353 | -0.000568 |
| 4 | exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h | 256 | 0.760017 | 0.758969 | 0.707252 | 0.503664 | 0.028406 | -0.014525 | 0.004088 |
| 5 | exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h | 1024 | 0.757201 | 0.758695 | 0.715957 | 0.506300 | 0.028132 | -0.005820 | 0.006725 |
| 6 | exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h | 512 | 0.757060 | 0.755967 | 0.703681 | 0.502472 | 0.025404 | -0.018096 | 0.002897 |
