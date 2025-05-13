library("MALDIquant")
library("MALDIquantForeign")
s=importTxt("/Users/apple/Desktop/PM/raw_PM")
# s=xxx是存放数据的文件夹地址


any(sapply(s,isEmpty)) #确认数据是否为空
table(sapply(s,length)) #确认数据点的数量和谱图的总量
all(sapply(s,isRegular)) #确认数据点间质量差是相等的或单调递增（MALDIquant才可用）
plot(s[[1]]) #绘制第一张谱图的图像并目视检查
plot(s[[2]]) #绘制第二张谱图的图像并目视检查
m=transformIntensity(s,method="sqrt") #使用平方根转化简化图形可视化并克服来自均值的方差的潜在依赖性
v=smoothIntensity(m,method="SavitzkyGolay",halfWindowSize=10) #使用Savitzky-Golay-Filter方法平滑谱图
baseline=estimateBaseline(v[[1]],method="SNIP",iterations=30) #使用SNIP算法获得基线，迭代次数为20
mixed=removeBaseline(v,method="SNIP",iterations=30) #去除基线
mixed2=calibrateIntensity(mixed, method="TIC") #为了克服批次效应使用TIC方法归一化
mixed3=trim(mixed2, range = c(100,1000)) #质量范围裁切
peaklists = lapply(mixed3, function(p) {
  cbind(as.matrix(p))
}) #返回一个矩阵
names(peaklists) = lapply(mixed, function(p) {
  metaData(p)$file
})

lapply(names(peaklists), function(f) {
  write.table(peaklists[[f]], paste0(f, '.preprocessed.txt'), row.names = FALSE)
})

# 可视化处理后的结果
# par(mfrow = c(2, 2))

# 绘制处理后的谱图
# for (i in 1:4) {
#   plot(mixed3[[i]], main = paste("Processed Spectrum", i))
# }

# 恢复原始的绘图设置
# par(mfrow = c(1, 1))
