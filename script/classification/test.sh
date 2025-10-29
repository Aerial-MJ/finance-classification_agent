# 零样本测试
python evaluate_text_image_k_fold.py \
 --checkpoint kfold_checkpoints/fold_3_best.pt \
 --image zlht.jpg \
 --text '借款合同(补偿贸易) 补偿贸易借款合同 立合同单位： （以下简称甲方） （以下简称乙方） 为明确责任，恪守信用，特订立本合同，共同遵守。 互协条件： 投资时间及金额： 投资时间共 年零月。自 年月日至 年月 日止。投资实际数额以支用凭证（并作为合同附件）分次或 一次支用。投资总额" 万元。其中甲方投资" 万元，乙'