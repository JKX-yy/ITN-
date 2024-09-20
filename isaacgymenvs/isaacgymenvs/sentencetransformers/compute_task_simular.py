
import os
from sentence_transformers import SentenceTransformer, util
    
def com_target_source_sim(target_task,source_task,module_path,target_task_save_path,similarity_threshold,target_actoin_seq,action_seq,ROOT_DIR):
    moddel_path_=os.path.join(ROOT_DIR, 'sentencetransformers/all-MiniLM-L6-v2')  
    model = SentenceTransformer(moddel_path_)
    # Encode all sentences
    embeddings2 = model.encode(target_task)
    embeddings_s=[]
    embeddings1 = model.encode(source_task)
 
    cos_sim = util.cos_sim(embeddings2, embeddings1)
    print(cos_sim.shape)
    print(cos_sim)
    res_source=[]
    res_module_path=[]
    for i in range(cos_sim.shape[1]):
        if cos_sim[0][i]>similarity_threshold:
            res_source.append(source_task[i])
            res_module_path.append(module_path[i])
            
    return cos_sim,target_task,res_source,res_module_path,target_task_save_path,target_actoin_seq,action_seq,ROOT_DIR

