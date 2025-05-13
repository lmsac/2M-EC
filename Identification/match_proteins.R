match.peaks = function(peaklist, reference, charge = 1, tolerance = 1000) {
  if (nrow(peaklist) > 0) {
    lapply(1:nrow(peaklist), function(i) {
      mz = as.numeric(peaklist[i, 1])
      mzRange = mz * c(1 - tolerance * 1e-6, 1, 1 + tolerance * 1e-6) #mz范围
      
      do.call(rbind, lapply(charge, function(ch) { #根据m/z范围和电荷状态计算出分子量的范围
        mwRange = (mzRange * ch) - ch
        rows = which(
          reference$MolecularWeight >= mwRange[1] & 
            reference$MolecularWeight <= mwRange[3] |
            reference$mintomax >= mwRange[1] & 
            reference$mintomax <= mwRange[3] |
            reference$Ntomax >= mwRange[1] & 
            reference$Ntomax <= mwRange[3] |
            reference$mintoC >= mwRange[1] & 
            reference$mintoC <= mwRange[3] 
        ) #找出参考数据中分子量落在计算出的范围内的行
        
        if (length(rows) > 0) {
          data.frame(
            MZ = mz,
            Intensity = as.numeric(peaklist[i, 2]),
            Charge = ch,
            MwDiff = mwRange[2] - reference$MolecularWeight[rows],
            MwDiffmintomax = mwRange[2] - reference$mintomax[rows],
            MwDiffNtomax = mwRange[2] - reference$Ntomax[rows],
            MwDiffmintoC = mwRange[2] - reference$mintoC[rows],
            reference[rows, ],
            stringsAsFactors = FALSE
          )
        }
        else {
          NULL
        }
      }))
    })
  }
  else {
    list()
  }
}


# 直接通过文件路径读取CSV和TXT文件
protein_file_path <- "/Users/apple/Desktop/feature selection/多电荷全长+碎片匹配/质量匹配.csv"
txt_file_path <- "//Users/apple/Desktop/feature selection/多电荷全长+碎片匹配/50feature.txt"  # 替换为你的TXT文件路径

# 读取指定的CSV文件
protein <- read.csv(protein_file_path)

# 读取指定的TXT文件
peaklist <- read.table(txt_file_path, header = TRUE)

# 匹配峰值，使用指定的蛋白质数据 (protein)
match_result <- do.call(
  rbind,
  match.peaks(
    peaklist,
    reference = protein,   # 使用从CSV读取的protein数据
    tolerance = 2000,
    charge = 1:3
  )
)

# 生成匹配结果的输出文件名，将.txt替换为.match.csv
output_file <- sub('\\.txt$', '.match.csv', txt_file_path)

# 保存匹配结果为CSV文件
write.csv(match_result, output_file, row.names = FALSE)
cat("匹配操作已完成，结果保存为", output_file, "\n")

