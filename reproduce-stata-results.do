import delimited using "data/raw/year_origin_hs92_4.tsv", delim("\t") clear
drop export_rca import_rca import_val
destring export_val year, replace force
drop if substr(origin,1,2)=="xx"
fillin year origin hs92
drop _fillin
ecomplexity export_val, i(origin) p(hs92) t(year) 
export delim using "data/raw/year_origin_hs92_4_ecomplexity_stata.csv", delim(",") quote replace
