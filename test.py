def remove_system_entries(input_list):
    return [entry for entry in input_list if entry.get('role') != 'system']

def update_persona(input_list):
    updated_list = remove_system_entries(input_list)
    updated_list.append({"role": "system", "content": "jhjhsdfkjh"})
    return updated_list


import pychatgpt as op

op.julia('ciao')
#%%

update_persona(op.chat_thread)