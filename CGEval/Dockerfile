FROM ubuntu:18.04

RUN apt-get update && apt-get install -y git perl wget unzip
RUN git clone https://github.com/INK-USC/CommonGen.git
RUN apt-get install -y build-essential libexpat1-dev libwww-perl libxml-perl libdb-dev

WORKDIR /tmp
ADD rouge.zip /tmp/rouge.zip
RUN unzip rouge.zip

WORKDIR /tmp/rouge

RUN tar -zxvf XML-Parser-2.44.tar.gz
RUN cd XML-Parser-2.44 && perl Makefile.PL && make && make test && make install

RUN tar -zxvf XML-RegExp-0.04.tar.gz 
RUN cd XML-RegExp-0.04 && perl Makefile.PL && make && make test && make install 

RUN tar -zxvf XML-DOM-1.46.tar.gz
RUN cd XML-DOM-1.46 && perl Makefile.PL && make && make test && make install  

RUN tar -zxvf DB_File-1.835.tar.gz
RUN cd DB_File-1.835 && perl Makefile.PL  && make && make test && make install

RUN tar -zxvf ROUGE-1.5.5.tgz
ENV ROUGE_EVAL_HOME "/tmp/rouge/RELEASE-1.5.5/data"
RUN cd RELEASE-1.5.5 && perl runROUGE-test.pl

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/anaconda3/ \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH "$PATH:/root/anaconda3/bin"

RUN conda create -n coco_score python=2.7
RUN /root/anaconda3/envs/coco_score/bin/pip install numpy networkx
RUN /root/anaconda3/envs/coco_score/bin/pip install -U spacy 
RUN /root/anaconda3/envs/coco_score/bin/python -m spacy download en_core_web_sm
RUN /root/anaconda3/envs/coco_score/bin/python -m spacy download en
RUN bash /CommonGen/evaluation/Traditional/get_stanford_models.sh

RUN conda create -n unilm python=3.6
RUN /root/anaconda3/envs/unilm/bin/pip install torch==1.4.0
RUN /root/anaconda3/envs/unilm/bin/pip install numpy

#RUN git clone https://github.com/NVIDIA/apex.git /tmp/apex && cd /tmp/apex && /root/anaconda3/envs/unilm/bin/python setup.py install --cpp_ext

RUN /root/anaconda3/envs/unilm/bin/pip install --user tensorboardX six numpy tqdm pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge nltk

RUN /root/anaconda3/envs/unilm/bin/python -c "import nltk; nltk.download('punkt')"
RUN /root/anaconda3/envs/unilm/bin/pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval
RUN pip install --editable /CommonGen/methods/unilm_based/unilm/src/

WORKDIR /root/
RUN ls
RUN mkdir -p /root/CommonGen-plus/dataset/final_data/commongen
ADD commongen_data/commongen /root/CommonGen-plus/dataset/final_data/commongen

RUN cp -r /tmp/rouge/src/nlg-eval/nlgeval/pycocoevalcap/meteor/* /CommonGen/evaluation/Traditional/eval_metrics/meteor/
RUN apt-get install -y openjdk-8-jdk openjdk-8-jre
RUN /root/anaconda3/envs/coco_score/bin/pip install psutil
RUN cd /CommonGen/evaluation/Traditional/ && /bin/sh get_stanford_models.sh

RUN git clone https://github.com/bheinzerling/pyrouge /tmp/pyrouge
RUN /root/anaconda3/envs/unilm/bin/pip install -e /tmp/pyrouge
RUN /root/anaconda3/envs/unilm/bin/python /tmp/pyrouge/bin/pyrouge_set_rouge_path "/tmp/rouge/RELEASE-1.5.5/"

RUN conda create -n pivot_score python=3.6
RUN /root/anaconda3/envs/pivot_score/bin/pip install spacy networkx
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN /root/anaconda3/envs/pivot_score/bin/python -m spacy download en
RUN /root/anaconda3/envs/pivot_score/bin/python -m spacy download en_core_web_sm

ADD run.sh /root/run.sh
RUN chmod +x /root/run.sh
RUN sed -i '54d' /CommonGen/evaluation/Traditional/eval_metrics/eval.py

ADD parse.py /root/parse.py
RUN chmod +x /root/parse.py

ENTRYPOINT ["/bin/sh", "-c", "./run.sh /pred.txt > out.txt 2>/dev/null && cat out.txt |./parse.py"]
