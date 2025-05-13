library(dplyr)
library(tidyr)
data <- read.csv("/Users/apple/Desktop/feature selection/20241009-export/20241010_Report_modified.csv")
data1 <- data
data2 <- group_by(data,Protein.Accession)  #分组并计算每个Protein.Accession的最小Start和最大End
data3 <- summarise(data2,min=min(Start))
data4 <- summarise(data2,max=max(End))
data5 <- full_join(data3,data4,by="Protein.Accession") #将两个结果合并
colnames(data5)<- c("ID","Min","Max") #重命名列名

library(Peptides)
library(dplyr)
library(tidyr)
ReadFasta<-function(file) {
  fasta<-readLines(file)
  ind<-grep(">", fasta)
  s<-data.frame(ind=ind, from=ind+1, to=c((ind-1)[-1], length(fasta)))
  seqs<-rep(NA, length(ind))
  for(i in 1:length(ind)) {
    seqs[i]<-paste(fasta[s$from[i]:s$to[i]], collapse="")
  }
  DF<-data.frame(name=gsub(">", "", fasta[ind]), sequence=seqs)
  return(DF)
}
seqs<-ReadFasta("/Users/apple/Desktop/feature selection/多分类单电荷匹配/output.fasta")
#读取fasta文件并将其转换为一个R数据框，每一行代表一个序列，包含序列名称和序列本身
database <- separate(seqs,name,into=c("sp","ID","other"),sep="[|]")  #将名称列拆分成三列
com <- left_join(data5,database,by="ID") #合并data5和database
result <- data.frame()
for (i in 1:nrow(com)){
  result[i,1]<- com$ID[i] #ID列复制到result数据框第一列
  result[i,2]<- mw(seq =substring(com$sequence[i],com$Min[i],com$Max[i]),monoisotopic = FALSE) #计算从Min到Max位置的序列片段分子量
  result[i,3]<- mw(seq =substring(com$sequence[i],1,com$Max[i]),monoisotopic = FALSE) #计算从序列开始到MAx位置的分子量
  result[i,4]<- mw(seq =substring(com$sequence[i],com$Min[i],nchar(com$sequence[i])),monoisotopic = FALSE) #计算从Min到序列结束的分子量
}
colnames(result)<- c("Protein","min-max","N-max","min-C")
result$Min <- com$Min
result$Max <- com$Max
write.csv(result,"/Users/apple/Desktop/proteinmass-重做.csv")

#一电荷匹配
library(dplyr)
data <- read.csv("/Users/apple/Desktop/feature selection/多分类单电荷匹配/proteinmass-重做-1.csv") 
peak <- read.delim("/Users/apple/Desktop/feature selection/POLE单电荷匹配/selected 50features-POLE.txt")
result <- data.frame()
for (i in 1:nrow(peak)){
  a <- filter(data,(data$min.max>=(peak[i,1]-peak[i,1]/500) & data$min.max<=(peak[i,1]+peak[i,1]/500))|(data$N.max>=(peak[i,1]-peak[i,1]/500) & data$N.max<=(peak[i,1]+peak[i,1]/500))|(data$min.C>=(peak[i,1]-peak[i,1]/500) & data$min.C<=(peak[i,1]+peak[i,1]/500))|(data$All>=(peak[i,1]-peak[i,1]/500) & data$All<=(peak[i,1]+peak[i,1]/500)))
  #筛选peaks值，条件为data中的所有列值在peaks的1/500内，也就是tolerance在2000ppm以内
  if ((is.na(a[1,1])==FALSE)){
    a$peak <- peak[i,1]
  }
  result <- rbind(result,a) #将peak值添加到reslut结果中
}
write.csv(result,"/Users/apple/Desktop/Pole50-一电荷匹配-1.csv")
