
def get_contexted(data_directory, window_size, data_split):
    #contexted = []
    #targets = []
    #personas = []
    train = []
    targets = []
    gold_labels = []
    toxicity = []
    #bad_dataset ={}
    with io.open(os.path.join(data_directory,data_split)) as f:
            total = 0
            count = 0

            for line in f:
                #tempList = []
                #tempList = list(filter(None,re.split("\t",line)))
                #print("tempList:", tempList)
                #print("listLen:", len(tempList))
                line = {temp.split(':', 1)[0].strip(): temp.split(':', 1)[1].strip() for temp in list(filter(None,re.split("\t",line.strip())))}
                #for k, v in line.items():
                #      print(k)
                text = line['text']
                utterances = text.split('\\n')
                #context = utterances[-(window_size+1):-1]
                context = '\n'.join(utterances[-(window_size+1):-1])
                target = utterances[-1]
                labels = line['labels']
                speaker_to_eval = line['speaker_to_eval']
                persona = '\n'.join(str(line['bot_persona']).split('\\n'))
                speaker_to_eval = line['speaker_to_eval']

                # if persona != 'nan':
                #   persona = persona.split(':',1)[1].replace('\\nyour persona:','').strip()
                
                total += 1

                if len(context) >= 1:   # Needs to have at least one context 
                    #contexted.append(''.join(utterances[-(window_size+1):-1]))
                    count += 1
                    train_sample = '\n'+persona+'\n'+ context                                            
                    label = target + '__end__'
                    target_sample = '__start__'+target 
                    gold_labels.append(label.strip())
                    train.append(train_sample.strip())
                    targets.append(target_sample.strip())
                    toxicity.append(labels.strip())
    #flatten = lambda l: [item for sublist in l for item in sublist]
    #contexted = flatten(contexted)             
    #return contexted, targets, personas
    return train , targets, gold_labels, toxicity #, total, count
