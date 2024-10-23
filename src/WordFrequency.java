import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;

public class WordFrequency {

    public static class WordMapper extends Mapper<Object, Text, Text, IntWritable> {
    private Set<String> stopWords = new HashSet<>();
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    // 加载停词表
    protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        Path stopWordsPath = new Path(conf.get("stopwords.path")); // 停词文件路径
        FileSystem fs = FileSystem.get(conf);  // 使用Hadoop FileSystem API来读取HDFS文件

        FSDataInputStream in = fs.open(stopWordsPath);  // 打开文件
        BufferedReader br = new BufferedReader(new InputStreamReader(in));  // 使用BufferedReader包裹流
        String stopWord;
        while ((stopWord = br.readLine()) != null) {
            stopWords.add(stopWord.trim().toLowerCase());
        }
        br.close();
        in.close();
    }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] fields = value.toString().split(",");
            if (fields.length == 4) {
                String headline = fields[1]; // 第二列为新闻标题
                // 去除标点，转换为小写，按空格分词
                String[] words = headline.replaceAll("[^a-zA-Z ]", "").toLowerCase().split("\\s+");

                for (String wordStr : words) {
                    // 忽略停词和空词
                    if (!stopWords.contains(wordStr) && !wordStr.isEmpty()) {
                        word.set(wordStr);
                        context.write(word, one); // 输出 (单词, 1)
                    }
                }
            }
        }
    }

    public static class WordReducer extends Reducer<Text, IntWritable, Text, Text> {
        private Map<String, Integer> wordCountMap = new HashMap<>();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            // 累加每个单词的出现次数
            for (IntWritable val : values) {
                sum += val.get();
            }
            // 将结果存入Map中
            wordCountMap.put(key.toString(), sum);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // 将Map转换为List并按出现次数进行降序排序
            List<Map.Entry<String, Integer>> sortedList = new ArrayList<>(wordCountMap.entrySet());
            Collections.sort(sortedList, new Comparator<Map.Entry<String, Integer>>() {
                public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                    return o2.getValue().compareTo(o1.getValue()); // 降序排序
                }
            });

            // 只输出前100个高频单词
            int rank = 1;
            for (Map.Entry<String, Integer> entry : sortedList) {
                if (rank > 100) break;
                String outputValue = String.format("%d: %s, %d", rank, entry.getKey(), entry.getValue());
                context.write(new Text(), new Text(outputValue)); // 输出格式为 "<排名>: <单词>, <次数>"
                rank++;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("stopwords.path", args[2]); // 停词表文件路径

        Job job = Job.getInstance(conf, "word frequency count");
        job.setJarByClass(WordFrequency.class);
        job.setMapperClass(WordMapper.class);
        job.setReducerClass(WordReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0])); // 输入文件路径
        FileOutputFormat.setOutputPath(job, new Path(args[1])); // 输出文件路径
        System.exit(job.waitForCompletion(true) ? 0 : 1); // 提交任务
    }
}