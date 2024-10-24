import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class StockCount {

    // Mapper类，用于统计股票代码的出现次数
    public static class StockMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text stockCode = new Text();

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String stock = extractStockCode(value.toString());
            if (!stock.isEmpty()) {
                stockCode.set(stock);
                context.write(stockCode, one); // 输出 (股票代码, 1)
            }
        }

        // 提取股票代码的方法，避免在多个地方处理字符串逻辑
        private String extractStockCode(String line) {
            String[] fields = line.split(",");
            return fields.length == 4 ? fields[3].trim() : "";
        }
    }

    // Reducer类，用于排序和输出结果
    public static class StockReducer extends Reducer<Text, IntWritable, Text, Text> {
        private List<StockCountPair> stockCounts = new ArrayList<>();

        // 内部类用于存储股票代码和计数
        public static class StockCountPair {
            private final String stock;
            private final int count;

            public StockCountPair(String stock, int count) {
                this.stock = stock;
                this.count = count;
            }

            public String getStock() {
                return stock;
            }

            public int getCount() {
                return count;
            }
        }

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            // 累加每个股票代码的出现次数
            for (IntWritable val : values) {
                sum += val.get();
            }
            // 将结果存入List中，稍后排序使用
            stockCounts.add(new StockCountPair(key.toString(), sum));
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // 使用Comparator.comparingInt进行降序排序
            stockCounts.sort(Comparator.comparingInt(StockCountPair::getCount).reversed());

            // 输出排名
            int rank = 1;
            for (StockCountPair pair : stockCounts) {
                context.write(new Text(), new Text(formatOutput(rank, pair)));
                rank++;
            }
        }

        // 提取格式化输出的公共逻辑
        private String formatOutput(int rank, StockCountPair pair) {
            return String.format("%d: %s, %d", rank, pair.getStock(), pair.getCount());
        }
    }

    // 主函数，配置并提交MapReduce作业
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "stock count");
        job.setJarByClass(StockCount.class);
        job.setMapperClass(StockMapper.class);
        job.setReducerClass(StockReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0])); // 输入文件路径
        FileOutputFormat.setOutputPath(job, new Path(args[1])); // 输出文件路径
        System.exit(job.waitForCompletion(true) ? 0 : 1); // 提交任务
    }
}