#!/usr/bin/env Rscript
# GO and KEGG Enrichment Analysis

library(clusterProfiler)
library(org.Mm.eg.db)

INPUT_DIR  <- "../results"
OUTPUT_DIR <- "../results"

up_genes <- readLines(file.path(INPUT_DIR, "DEG_up.txt"))
down_genes <- readLines(file.path(INPUT_DIR, "DEG_down.txt"))

up_entrez <- bitr(up_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Mm.eg.db)
down_entrez <- bitr(down_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Mm.eg.db)

# GO enrichment
ego_up <- enrichGO(gene = up_entrez$ENTREZID, OrgDb = org.Mm.eg.db, ont = "ALL",
                   pAdjustMethod = "BH", pvalueCutoff = 0.05, qvalueCutoff = 0.05, readable = TRUE)
ego_down <- enrichGO(gene = down_entrez$ENTREZID, OrgDb = org.Mm.eg.db, ont = "ALL",
                     pAdjustMethod = "BH", pvalueCutoff = 0.05, qvalueCutoff = 0.05, readable = TRUE)

if (!is.null(ego_up) && nrow(ego_up@result) > 0)
    write.csv(ego_up@result, file.path(OUTPUT_DIR, "GO_up.csv"), row.names = FALSE)
if (!is.null(ego_down) && nrow(ego_down@result) > 0)
    write.csv(ego_down@result, file.path(OUTPUT_DIR, "GO_down.csv"), row.names = FALSE)

# KEGG enrichment
ekegg_up <- enrichKEGG(gene = up_entrez$ENTREZID, organism = "mmu",
                       pAdjustMethod = "BH", pvalueCutoff = 0.05, qvalueCutoff = 0.05)
ekegg_down <- enrichKEGG(gene = down_entrez$ENTREZID, organism = "mmu",
                         pAdjustMethod = "BH", pvalueCutoff = 0.05, qvalueCutoff = 0.05)

if (!is.null(ekegg_up) && nrow(ekegg_up@result) > 0)
    write.csv(ekegg_up@result, file.path(OUTPUT_DIR, "KEGG_up.csv"), row.names = FALSE)
if (!is.null(ekegg_down) && nrow(ekegg_down@result) > 0)
    write.csv(ekegg_down@result, file.path(OUTPUT_DIR, "KEGG_down.csv"), row.names = FALSE)

cat("Done\n")
