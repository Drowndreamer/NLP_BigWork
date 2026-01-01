
import pandas as pd
import numpy as np
import torch
import jieba
import jieba.posseg as pseg
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
import json
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存目录
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/models', exist_ok=True)

print("="*80)
print("欺诈对话检测对抗攻击完整实验 - 高配置优化版")
print("="*80)

# ============================================
# 1. 数据加载与预处理
# ============================================

class FraudDataProcessor:
    """欺诈对话数据处理器"""
    
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        
    def load_data(self):
        """加载训练集和测试集"""
        print("\n[1/6] 正在加载数据...")
        self.train_df = pd.read_csv(self.train_path, encoding='utf-8')
        self.test_df = pd.read_csv(self.test_path, encoding='utf-8')
        
        print(f"✓ 训练集大小: {len(self.train_df)}")
        print(f"✓ 测试集大小: {len(self.test_df)}")
        
        # 数据清洗
        self.train_df = self.train_df.dropna(subset=['specific_dialogue_content', 'is_fraud'])
        self.test_df = self.test_df.dropna(subset=['specific_dialogue_content', 'is_fraud'])
        
        return self.train_df, self.test_df
    
    def preprocess_text(self, text):
        """文本预处理：中文分词"""
        if not isinstance(text, str):
            return ""
        # 使用jieba分词
        words = jieba.cut(text.strip())
        return ' '.join(words)
    
    def prepare_data(self):
        """准备训练和测试数据"""
        self.train_df, self.test_df = self.load_data()
        
        # 文本预处理
        print("✓ 正在进行中文分词...")
        self.train_df['processed_text'] = self.train_df['specific_dialogue_content'].apply(
            self.preprocess_text
        )
        self.test_df['processed_text'] = self.test_df['specific_dialogue_content'].apply(
            self.preprocess_text
        )
        
        # 标签处理
        self.train_df['label'] = self.train_df['is_fraud'].astype(int)
        self.test_df['label'] = self.test_df['is_fraud'].astype(int)
        
        # 统计信息
        print(f"✓ 训练集 - 欺诈: {self.train_df['label'].sum()}, 正常: {len(self.train_df) - self.train_df['label'].sum()}")
        print(f"✓ 测试集 - 欺诈: {self.test_df['label'].sum()}, 正常: {len(self.test_df) - self.test_df['label'].sum()}")
        
        return self.train_df, self.test_df

# ============================================
# 2. 逻辑回归分类器
# ============================================

class LogisticRegressionClassifier:
    """逻辑回归分类器"""
    
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # 处理类别不平衡
        )
    
    def train(self, train_texts, train_labels):
        """训练模型"""
        print("\n[2/6] 训练逻辑回归模型...")
        
        # 特征提取
        X_train = self.vectorizer.fit_transform(train_texts)
        print(f"✓ 特征维度: {X_train.shape}")
        
        # 训练
        self.model.fit(X_train, train_labels)
        
        # 训练集准确率
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(train_labels, train_pred)
        print(f"✓ 训练集准确率: {train_acc:.4f}")
        
        return self.model
    
    def evaluate(self, test_texts, test_labels):
        """评估模型"""
        X_test = self.vectorizer.transform(test_texts)
        predictions = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)
        
        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='binary', zero_division=0
        )
        
        results = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'probabilities': probs
        }
        
        return results
    
    def predict_proba(self, texts):
        """预测概率"""
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

# ============================================
# 3. BERT分类器（优化版）
# ============================================

class BERTClassifier:
    """BERT分类器 - 优化版"""
    
    def __init__(self, model_name='bert-base-chinese', max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ 使用设备: {self.device}")
        
        # 加载tokenizer和模型
        print(f"✓ 加载预训练模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)
    
    def train(self, train_texts, train_labels, epochs=3, batch_size=32, learning_rate=2e-5):
        """训练模型"""
        print("\n[3/6] 训练BERT模型...")
        
        # Tokenization
        encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 创建数据集
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(train_labels)
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # 优化器和学习率调度
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                # 计算准确率
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}'
                })
            
            avg_loss = total_loss / len(train_loader)
            epoch_acc = correct / total
            print(f"✓ Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={epoch_acc:.4f}")
        
        return self.model
    
    def evaluate(self, test_texts, test_labels, batch_size=32):
        """评估模型"""
        encodings = self.tokenizer(
            test_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(test_labels)
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估BERT"):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        acc = accuracy_score(test_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, all_preds, average='binary', zero_division=0
        )
        
        results = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs)
        }
        
        return results
    
    def predict_proba(self, texts, batch_size=32):
        """预测概率"""
        if isinstance(texts, str):
            texts = [texts]
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)

# ============================================
# 4. 增强版TextFooler对抗攻击
# ============================================

class EnhancedTextFooler:
    """增强版TextFooler对抗攻击"""
    
    def __init__(self, model, model_type='lr', similarity_threshold=0.84):
        self.model = model
        self.model_type = model_type
        self.similarity_threshold = similarity_threshold
        
        # 超大规模中文同义词词典
        self.synonym_dict = {
            # 金融相关 (40+词)
            '转账': ['汇款', '打款', '转钱', '付款', '支付', '转入', '转出', '划款'],
            '银行卡': ['卡号', '账号', '账户', '银行账户', '储蓄卡', '银行卡号'],
            '密码': ['口令', '密钥', '验证码', '动态码', '支付密码', '交易密码'],
            '身份证': ['证件', '身份信息', 'ID', '身份证号', '证件号', '身份证件'],
            '钱': ['资金', '款项', '费用', '金额', '现金', '货币'],
            '账户': ['账号', '户头', '帐号', '银行账户', '账户信息'],
            '贷款': ['借款', '借贷', '融资', '借钱'],
            '还款': ['偿还', '归还', '付款', '清偿'],
            '利息': ['利率', '息金', '利钱'],
            '信用卡': ['信用卡片', '贷记卡', '信用卡账户'],
            
            # 动作相关 (50+词)
            '验证': ['核实', '确认', '校验', '审核', '检验', '核对', '查验'],
            '提供': ['给出', '告知', '发送', '提交', '上传', '出示'],
            '办理': ['处理', '操作', '执行', '申请', '开通', '受理'],
            '联系': ['联络', '沟通', '找', '咨询', '对接', '联系上'],
            '告诉': ['告知', '说', '讲', '通知', '说明', '表示'],
            '输入': ['填写', '录入', '输入进', '填入', '键入'],
            '点击': ['点', '按', '选择', '打开', '按下'],
            '下载': ['下载安装', '安装', '获取', '下'],
            '注册': ['登记', '注册账号', '开户', '申请'],
            '登录': ['登陆', '登入', '进入', '访问'],
            '发送': ['发', '传', '传送', '寄'],
            '接收': ['收到', '接到', '得到', '获得'],
            '查询': ['查看', '查', '检查', '核查'],
            '修改': ['更改', '变更', '调整', '改'],
            
            # 时间相关 (30+词)
            '立即': ['马上', '迅速', '赶快', '立刻', '快速', '尽快', '即刻'],
            '现在': ['目前', '当前', '此刻', '眼下', '这会儿', '现时'],
            '今天': ['今日', '本日', '当天', '今'],
            '明天': ['明日', '次日', '第二天'],
            '马上': ['立刻', '立即', '赶紧', '立马'],
            '尽快': ['尽早', '尽速', '速速', '快点'],
            '及时': ['迅速', '快速', '实时', '即时'],
            
            # 安全相关 (20+词)
            '安全': ['可靠', '保险', '稳妥', '保障', '放心', '安心'],
            '保护': ['保障', '维护', '守护', '确保', '保卫'],
            '风险': ['危险', '隐患', '问题', '威胁', '隐忧'],
            '冻结': ['冻结账户', '锁定', '封', '停用'],
            '解冻': ['解锁', '恢复', '启用', '开通'],
            
            # 身份相关 (30+词)
            '客服': ['服务人员', '工作人员', '客户经理', '专员', '客服人员'],
            '警察': ['民警', '公安', '警方', '执法人员', '警官'],
            '工作人员': ['职员', '员工', '人员', '专员', '工作者'],
            '法院': ['法庭', '司法机关', '人民法院'],
            '银行': ['银行机构', '金融机构', '银行单位'],
            '公司': ['企业', '机构', '单位', '公司单位'],
            
            # 需求相关 (25+词)
            '需要': ['必须', '要求', '应当', '得', '要', '需'],
            '帮助': ['协助', '帮忙', '辅助', '支持', '帮'],
            '配合': ['协助', '帮忙', '支持', '参与', '协同'],
            '允许': ['同意', '许可', '批准', '准许'],
            '拒绝': ['不同意', '否决', '驳回', '不允许'],
            
            # 信息相关 (30+词)
            '信息': ['资料', '内容', '数据', '情况', '详情', '材料'],
            '个人': ['私人', '自己的', '本人的', '您的', '个人的'],
            '资料': ['信息', '材料', '内容', '数据', '档案'],
            '详情': ['详细信息', '具体情况', '细节', '详细资料'],
            '号码': ['号', '号数', '编号', '序号'],
            
            # 程度相关 (25+词)
            '非常': ['很', '特别', '十分', '极其', '相当', '格外'],
            '可能': ['或许', '也许', '大概', '可以', '兴许'],
            '必须': ['一定', '务必', '需要', '得', '肯定要'],
            '一定': ['必定', '肯定', '必然', '定'],
            '绝对': ['肯定', '一定', '必定', '铁定'],
            
            # 描述相关 (30+词)
            '重要': ['关键', '要紧', '紧要', '重大', '要害'],
            '紧急': ['着急', '急迫', '火急', '要紧', '紧要'],
            '免费': ['不收费', '无偿', '零费用', '免费用', '不要钱'],
            '真实': ['真的', '确实', '属实', '真正'],
            '虚假': ['假的', '不实', '虚构', '编造'],
            '正常': ['正规', '合法', '常规', '正当'],
            '异常': ['不正常', '反常', '特殊', '奇怪'],
            
            # 常见动词 (40+词)
            '进行': ['开展', '实施', '执行', '做', '进行中'],
            '完成': ['搞定', '做完', '办好', '处理好', '完结'],
            '操作': ['处理', '办理', '执行', '弄', '操控'],
            '使用': ['用', '运用', '使', '采用'],
            '退出': ['退', '离开', '退出登录', '退出系统'],
            '保存': ['存储', '存', '储存', '保留'],
            '删除': ['删', '移除', '去掉', '清除'],
            '取消': ['撤销', '作废', '废除', '取消掉'],
            '确定': ['确认', '肯定', '确定好'],
            
            # 电商/营销相关 (30+词)
            '优惠': ['折扣', '特价', '打折', '便宜', '优惠活动'],
            '活动': ['促销', '优惠', '特卖', '推广', '活动'],
            '中奖': ['获奖', '中彩', '得奖', '赢得', '抽中'],
            '奖品': ['奖励', '礼品', '奖金', '赠品', '奖项'],
            '购买': ['买', '购', '购入', '采购'],
            '订单': ['订购', '定单', '购买记录'],
            '退款': ['退钱', '返款', '退费', '退还'],
            '快递': ['快件', '包裹', '邮件', '快递包裹'],
            
            # 诱导相关 (25+词)
            '赶紧': ['赶快', '快点', '抓紧', '尽快', '速速'],
            '仅限': ['只有', '限定', '唯一', '专属', '仅'],
            '机会': ['时机', '良机', '契机', '可能性', '机遇'],
            '限时': ['有限时间', '时间有限', '限期'],
            '独家': ['专属', '独有', '唯一', '特有'],
            
            # 否定词 (15+词)
            '不': ['没', '不会', '不要', '别'],
            '没有': ['没', '无', '不存在'],
            '不能': ['不可以', '不行', '无法', '不得'],
            '无法': ['不能', '没办法', '不可能'],
        }
    
    def get_synonyms(self, word):
        """获取同义词"""
        return self.synonym_dict.get(word, [])
    
    def word_importance(self, text, label):
        """计算词重要性（优化版）"""
        words = text.split()
        if len(words) == 0:
            return []
        
        importances = []
        
        try:
            # 原始预测
            original_probs = self.model.predict_proba([text])[0]
            if len(original_probs) <= label:
                return []
            original_prob = original_probs[label]
            
            for i, word in enumerate(words):
                # 跳过单字符词（标点、停用词）
                if len(word) <= 1:
                    importances.append((i, word, 0.0))
                    continue
                
                # 删除该词后的文本
                temp_words = words[:i] + words[i+1:]
                temp_text = ' '.join(temp_words)
                
                if temp_text.strip():
                    try:
                        new_probs = self.model.predict_proba([temp_text])[0]
                        if len(new_probs) > label:
                            new_prob = new_probs[label]
                            importance = original_prob - new_prob
                        else:
                            importance = 0.0
                    except Exception as e:
                        importance = 0.0
                else:
                    importance = 0.0
                
                importances.append((i, word, importance))
        
        except Exception as e:
            print(f"计算重要性出错: {e}")
            importances = [(i, word, 0.0) for i, word in enumerate(words)]
        
        # 按重要性降序排序
        importances.sort(key=lambda x: x[2], reverse=True)
        return importances
    
    def attack(self, text, original_label, max_replacements=15):
        """执行对抗攻击"""
        words = text.split()
        
        if len(words) == 0:
            return {
                'success': False,
                'adversarial_text': text,
                'original_text': text,
                'num_replacements': 0,
                'replaced_words': [],
                'new_prediction': original_label,
                'new_probability': 0.5
            }
        
        # 计算词重要性
        importances = self.word_importance(text, original_label)
        
        # 执行替换
        modified_words = words.copy()
        replacements = 0
        replaced_words = []
        
        for idx, word, importance in importances:
            if replacements >= max_replacements:
                break
            
            if idx >= len(modified_words):
                continue
            
            # 获取同义词
            synonyms = self.get_synonyms(word)
            if not synonyms:
                continue
            
            # 尝试每个同义词
            best_syn = None
            best_prob = float('inf')
            
            for syn in synonyms:
                if syn == word:
                    continue
                
                temp_words = modified_words.copy()
                temp_words[idx] = syn
                temp_text = ' '.join(temp_words)
                
                try:
                    new_probs = self.model.predict_proba([temp_text])[0]
                    
                    if len(new_probs) <= 1:
                        continue
                    
                    new_pred = int(np.argmax(new_probs))
                    
                    # 攻击成功，立即返回
                    if new_pred != original_label:
                        return {
                            'success': True,
                            'adversarial_text': temp_text,
                            'original_text': text,
                            'num_replacements': replacements + 1,
                            'replaced_words': replaced_words + [f'{word}→{syn}'],
                            'new_prediction': new_pred,
                            'new_probability': float(new_probs[new_pred])
                        }
                    
                    # 记录最佳替换
                    if new_probs[original_label] < best_prob:
                        best_prob = new_probs[original_label]
                        best_syn = syn
                
                except:
                    continue
            
            # 使用最佳替换
            if best_syn is not None:
                modified_words[idx] = best_syn
                replaced_words.append(f'{word}→{best_syn}')
                replacements += 1
        
        # 最终检查
        final_text = ' '.join(modified_words)
        try:
            final_probs = self.model.predict_proba([final_text])[0]
            final_pred = int(np.argmax(final_probs))
            
            return {
                'success': final_pred != original_label,
                'adversarial_text': final_text,
                'original_text': text,
                'num_replacements': replacements,
                'replaced_words': replaced_words,
                'new_prediction': final_pred,
                'new_probability': float(final_probs[final_pred])
            }
        except:
            return {
                'success': False,
                'adversarial_text': final_text,
                'original_text': text,
                'num_replacements': replacements,
                'replaced_words': replaced_words,
                'new_prediction': original_label,
                'new_probability': 0.5
            }

# ============================================
# 5. 评估与可视化
# ============================================

def print_results(results, dataset_name=""):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} 评估结果")
    print(f"{'='*60}")
    print(f"准确率 (Accuracy):  {results['accuracy']:.4f}")
    print(f"精确率 (Precision): {results['precision']:.4f}")
    print(f"召回率 (Recall):    {results['recall']:.4f}")
    print(f"F1分数 (F1-Score):  {results['f1']:.4f}")
    print(f"{'='*60}\n")

def plot_confusion_matrix(y_true, y_pred, title="混淆矩阵", save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '欺诈'],
                yticklabels=['正常', '欺诈'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ============================================
# 6. 主实验流程
# ============================================

def main():
    """主实验流程 - 高配置优化版"""
    
    print("\n" + "="*80)
    print("开始完整实验")
    print("="*80)
    
    # 1. 数据准备
    processor = FraudDataProcessor('训练集结果.csv', '测试集结果.csv')
    train_df, test_df = processor.prepare_data()
    
    train_texts = train_df['processed_text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['processed_text'].tolist()
    test_labels = test_df['label'].tolist()
    
    # 2. 训练逻辑回归
    lr_model = LogisticRegressionClassifier(max_features=10000)
    lr_model.train(train_texts, train_labels)
    
    # 评估逻辑回归
    lr_results = lr_model.evaluate(test_texts, test_labels)
    print_results(lr_results, "逻辑回归 - 原始测试集")
    plot_confusion_matrix(test_labels, lr_results['predictions'],
                         "逻辑回归混淆矩阵",
                         'results/figures/lr_confusion_matrix.png')
    
    # 3. 训练BERT
    use_bert = torch.cuda.is_available()
    print(f"\n{'='*60}")
    print(f"GPU可用: {use_bert}")
    print(f"{'='*60}")
    
    if use_bert:
        bert_model = BERTClassifier(model_name='bert-base-chinese')
        bert_model.train(train_texts, train_labels, epochs=3, batch_size=32)
        
        bert_results = bert_model.evaluate(test_texts, test_labels)
        print_results(bert_results, "BERT - 原始测试集")
        plot_confusion_matrix(test_labels, bert_results['predictions'],
                             "BERT混淆矩阵",
                             'results/figures/bert_confusion_matrix.png')
    
    # 4. 对抗攻击 - 逻辑回归
    print("\n" + "="*80)
    print("[4/6] 对逻辑回归执行对抗攻击")
    print("="*80)
    
    lr_attacker = EnhancedTextFooler(lr_model, model_type='lr')
    
    # 只攻击欺诈样本
    fraud_mask = [label == 1 for label in test_labels]
    fraud_texts = [text for text, is_fraud in zip(test_texts, fraud_mask) if is_fraud]
    fraud_labels = [label for label, is_fraud in zip(test_labels, fraud_mask) if is_fraud]
    
    print(f"✓ 找到 {len(fraud_texts)} 个欺诈样本")
    
    # 攻击所有欺诈样本（或前100个）
    attack_count = min(100, len(fraud_texts))
    print(f"✓ 将攻击前 {attack_count} 个欺诈样本")
    
    lr_attack_results = []
    for i in tqdm(range(attack_count), desc="攻击LR"):
        result = lr_attacker.attack(fraud_texts[i], 1, max_replacements=15)
        lr_attack_results.append(result)
    
    lr_asr = sum([r['success'] for r in lr_attack_results]) / len(lr_attack_results)
    lr_avg_pert = np.mean([r['num_replacements'] for r in lr_attack_results])
    
    print(f"\n✓ 逻辑回归攻击成功率 (ASR): {lr_asr:.2%}")
    print(f"✓ 平均替换词数: {lr_avg_pert:.2f}")
    
    # 评估对抗样本
    adv_texts_lr = [r['adversarial_text'] for r in lr_attack_results]
    adv_labels = fraud_labels[:attack_count]
    lr_adv_results = lr_model.evaluate(adv_texts_lr, adv_labels)
    print_results(lr_adv_results, "逻辑回归 - 对抗测试集")
    
    # 5. 对抗攻击 - BERT
    if use_bert:
        print("\n" + "="*80)
        print("[5/6] 对BERT执行对抗攻击")
        print("="*80)
        
        bert_attacker = EnhancedTextFooler(bert_model, model_type='bert')
        
        bert_attack_results = []
        for i in tqdm(range(attack_count), desc="攻击BERT"):
            result = bert_attacker.attack(fraud_texts[i], 1, max_replacements=15)
            bert_attack_results.append(result)
        
        bert_asr = sum([r['success'] for r in bert_attack_results]) / len(bert_attack_results)
        bert_avg_pert = np.mean([r['num_replacements'] for r in bert_attack_results])
        
        print(f"\n✓ BERT攻击成功率 (ASR): {bert_asr:.2%}")
        print(f"✓ 平均替换词数: {bert_avg_pert:.2f}")
        
        adv_texts_bert = [r['adversarial_text'] for r in bert_attack_results]
        bert_adv_results = bert_model.evaluate(adv_texts_bert, adv_labels)
        print_results(bert_adv_results, "BERT - 对抗测试集")
    
    # 6. 结果汇总
    print("\n" + "="*80)
    print("[6/6] 结果汇总与保存")
    print("="*80)
    
    summary = {
        'LR原始准确率': lr_results['accuracy'],
        'LR对抗准确率': lr_adv_results['accuracy'],
        'LR攻击成功率': lr_asr,
        'LR平均扰动率': lr_avg_pert / len(fraud_texts[0].split()) if fraud_texts else 0,
    }
    
    if use_bert:
        summary.update({
            'BERT原始准确率': bert_results['accuracy'],
            'BERT对抗准确率': bert_adv_results['accuracy'],
            'BERT攻击成功率': bert_asr,
            'BERT平均扰动率': bert_avg_pert / len(fraud_texts[0].split()) if fraud_texts else 0,
        })
    
    # 保存结果
    pd.DataFrame([summary]).to_csv('results/实验结果汇总.csv', index=False, encoding='utf-8')
    
    # 保存对抗样本
    lr_samples_df = pd.DataFrame([
        {
            'original_text': r['original_text'],
            'adversarial_text': r['adversarial_text'],
            'success': r['success'],
            'num_replacements': r['num_replacements'],
            'replaced_words': ', '.join(r['replaced_words'])
        }
        for r in lr_attack_results
    ])
    lr_samples_df.to_csv('results/LR对抗样本.csv', index=False, encoding='utf-8')
    
    if use_bert:
        bert_samples_df = pd.DataFrame([
            {
                'original_text': r['original_text'],
                'adversarial_text': r['adversarial_text'],
                'success': r['success'],
                'num_replacements': r['num_replacements'],
                'replaced_words': ', '.join(r['replaced_words'])
            }
            for r in bert_attack_results
        ])
        bert_samples_df.to_csv('results/BERT对抗样本.csv', index=False, encoding='utf-8')
    
    # 绘制攻击效果对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if use_bert:
        models = ['逻辑回归', 'BERT']
        original = [lr_results['accuracy'], bert_results['accuracy']]
        adversarial = [lr_adv_results['accuracy'], bert_adv_results['accuracy']]
    else:
        models = ['逻辑回归']
        original = [lr_results['accuracy']]
        adversarial = [lr_adv_results['accuracy']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, original, width, label='原始准确率', color='#2E86AB')
    bars2 = ax.bar(x + width/2, adversarial, width, label='对抗准确率', color='#A23B72')
    
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('对抗攻击效果对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/figures/攻击效果对比.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印最终总结
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print("\n结果已保存到 results/ 目录:")
    print("  - 实验结果汇总.csv")
    print("  - LR对抗样本.csv")
    if use_bert:
        print("  - BERT对抗样本.csv")
    print("  - figures/ (所有图表)")
    
    print("\n最终结果:")
    for key, value in summary.items():
        if '率' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 展示成功案例
    print("\n" + "="*80)
    print("对抗攻击成功案例")
    print("="*80)
    
    successful_lr = [r for r in lr_attack_results if r['success']]
    for i, result in enumerate(successful_lr[:3]):
        print(f"\n【案例 {i+1}】")
        print(f"原文: {result['original_text'][:100]}...")
        print(f"对抗: {result['adversarial_text'][:100]}...")
        print(f"替换: {', '.join(result['replaced_words'][:5])}")
        print("-" * 80)

if __name__ == "__main__":
    main()