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
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            Path stopWordsPath = new Path(conf.get("stopwords.path")); // 停词文件路径
            FileSystem fs = FileSystem.get(conf);
            loadStopWords(fs, stopWordsPath);
        }

        // 分离加载停词表的逻辑
        private void loadStopWords(FileSystem fs, Path stopWordsPath) throws IOException {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(stopWordsPath)))) {
                String stopWord;
                while ((stopWord = br.readLine()) != null) {
                    stopWords.add(stopWord.trim().toLowerCase());
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String headline = extractHeadline(value.toString());
            if (headline != null) {
                String[] words = tokenizeAndNormalize(headline); // 标准化并分词
                for (String wordStr : words) {
                    if (isValidWord(wordStr)) {
                        word.set(wordStr);
                        context.write(word, one); // 输出 (单词, 1)
                    }
                }
            }
        }

        // 提取新闻标题
        private String extractHeadline(String line) {
            String[] fields = line.split(",");
            return fields.length == 4 ? fields[1].trim() : null;
        }

        // 分词并去除标点符号和大小写
        private String[] tokenizeAndNormalize(String headline) {
            return headline.replaceAll("[^a-zA-Z ]", "").toLowerCase().split("\\s+");
        }

        // 检查单词是否有效
        private boolean isValidWord(String word) {
            return !stopWords.contains(word) && !word.isEmpty();
        }
    }

    public static class WordReducer extends Reducer<Text, IntWritable, Text, Text> {
        private Map<String, Integer> wordCountMap = new HashMap<>();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            wordCountMap.merge(key.toString(), sumValues(values), Integer::sum); // 直接使用Map合并
        }

        // 累加每个单词的出现次数
        private int sumValues(Iterable<IntWritable> values) {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            return sum;
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            List<Map.Entry<String, Integer>> sortedList = getSortedWordList(); // 排序
            outputTopWords(context, sortedList, 100); // 输出前100个
        }

        // 排序Map中的词频
        private List<Map.Entry<String, Integer>> getSortedWordList() {
            return wordCountMap.entrySet()
                    .stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed()) // 按值降序排序
                    .toList();
        }

        // 输出前N个高频词
        private void outputTopWords(Context context, List<Map.Entry<String, Integer>> sortedList, int limit) throws IOException, InterruptedException {
            for (int i = 0; i < Math.min(limit, sortedList.size()); i++) {
                Map.Entry<String, Integer> entry = sortedList.get(i);
                String outputValue = String.format("%d: %s, %d", i + 1, entry.getKey(), entry.getValue());
                context.write(new Text(), new Text(outputValue)); // 输出格式 "<排名>: <单词>, <次数>"
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