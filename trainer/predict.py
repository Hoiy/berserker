import tensorflow as tf
from berserker.transform import batch_preprocess, batch_postprocessing
import numpy as np
import pandas as pd

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

def main(_):
    # texts = pd.read_csv(FLAGS.predict_file, header=None, sep='^')[0]
    texts = [
        '共同创造美好的新世纪——二○○一年新年贺词',
        '（二○○○年十二月三十一日）（附图片1张）',
        '女士们，先生们，同志们，朋友们：',
        '2001年新年钟声即将敲响。人类社会前进的航船就要驶入21世纪的新航程。中国人民进入了向现代化建设第三步战略目标迈进的新征程。',
        '在这个激动人心的时刻，我很高兴通过中国国际广播电台、中央人民广播电台和中央电视台，向全国各族人民，向香港特别行政区同胞、澳门特别行政区同胞和台湾同胞、海外侨胞，向世界各国的朋友们，致以新世纪第一个新年的祝贺！',
        '过去的一年，是我国社会主义改革开放和现代化建设进程中具有标志意义的一年。在中国共产党的领导下，全国各族人民团结奋斗，国民经济继续保持较快的发展势头，经济结构的战略性调整顺利部署实施。西部大开发取得良好开端。精神文明建设和民主法制建设进一步加强。我们在过去几年取得成绩的基础上，胜利完成了第九个五年计划。我国已进入了全面建设小康社会，加快社会主义现代化建设的新的发展阶段。',
        '面对新世纪，世界各国人民的共同愿望是：继续发展人类以往创造的一切文明成果，克服20世纪困扰着人类的战争和贫困问题，推进和平与发展的崇高事业，创造一个美好的世界。',
        '我们希望，新世纪成为各国人民共享和平的世纪。在20世纪里，世界饱受各种战争和冲突的苦难。时至今日，仍有不少国家和地区的人民还在忍受战火的煎熬。中国人民真诚地祝愿他们早日过上和平安定的生活。中国人民热爱和平与自由，始终奉行独立自主的和平外交政策，永远站在人类正义事业的一边。我们愿同世界上一切爱好和平的国家和人民一道，为促进世界多极化，建立和平稳定、公正合理的国际政治经济新秩序而努力奋斗。' * 10
    ]

    bert_inputs, mappings, sizes = batch_preprocess(
        texts,
        FLAGS.max_seq_length
    )

    berserker = tf.contrib.predictor.from_saved_model(
        '/tmp/export/1547563491'
    )
    bert_outputs = berserker(bert_inputs)
    
    results = batch_postprocessing(
        texts,
        mappings,
        sizes,
        bert_inputs,
        bert_outputs,
        FLAGS.max_seq_length
    )
    print(results)



if __name__ == "__main__":
  tf.app.run()
