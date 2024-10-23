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

    public static class StockMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text stockCode = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // 按逗号分隔行数据
            String[] fields = value.toString().split(",");
            // 确保有至少4列
            if (fields.length == 4) {
                String stock = fields[3].trim(); // 第四列为股票代码
                if (!stock.isEmpty()) {
                    stockCode.set(stock);
                    context.write(stockCode, one); // 输出 (股票代码, 1)
                }
            }
        }
    }

    public static class StockReducer extends Reducer<Text, IntWritable, Text, Text> {
        private List<StockCountPair> stockCounts = new ArrayList<>();

        // 内部类用于存储股票代码和计数
        public static class StockCountPair {
            private String stock;
            private int count;

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
            // 对stockCounts按出现次数进行降序排序
            Collections.sort(stockCounts, new Comparator<StockCountPair>() {
                public int compare(StockCountPair o1, StockCountPair o2) {
                    return o2.getCount() - o1.getCount(); // 降序排序
                }
            });

            // 输出排名
            int rank = 1;
            for (StockCountPair pair : stockCounts) {
                String outputValue = String.format("%d: %s, %d", rank, pair.getStock(), pair.getCount());
                context.write(new Text(), new Text(outputValue)); // 输出格式为 "<排名>：<股票代码>，<次数>"
                rank++;
            }
        }
    }

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