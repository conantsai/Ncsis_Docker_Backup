import os
import docker
import time
import tarfile
import shutil

def full_backup(container, id, conimg, time_stamp, fb_path):
    # export the container to tarfile
    container_tararchive = container.export()
    container_tar = fb_path + id + "_" + conimg + "_" + time_stamp + ".tar"
    with open(container_tar, mode = 'wb') as img_tar:
        for chunk in container_tararchive:
            img_tar.write(chunk)
        
    return 

def fs_difference(diff_cmd, diff_list):
    cmd = os.popen(diff_cmd)
    result = cmd.readlines()
    cmd.close()
    for i in result:
        if i != " " and i != "None":
            i = i[:-1]
            diff_list.append(i)

    return diff_list

def copy_alldir(dir_arc, dir_IB, ib_path):
    # copy file architecture 
    container_dir = ib_path + dir_arc
    folder_list = ib_path + "datadir.list"
    folder_sh = ib_path + "datadir.sh"
    os.system("find " + container_dir + " -type d > " + folder_list)
    os.system("cat " + folder_list + " | sed \'s/^/mkdir -p /\' | sed 's/" +dir_arc + "/" + dir_IB + "/\' | sed \'s/"+ dir_IB + "/&\/Add/g\' > " + folder_sh)
    os.system("sh " + folder_sh)
    os.system("cat " + folder_list + " | sed \'s/^/mkdir -p /\' | sed 's/" +dir_arc + "/" + dir_IB + "/\' | sed \'s/"+ dir_IB + "/&\/Modify/g\' > " + folder_sh)
    os.system("sh " + folder_sh)
    
    return folder_list, folder_sh

def word_position(string, subStr, findCnt):
    listStr = string.split(subStr,findCnt)
    if len(listStr) <= findCnt:
        return "not find"
    return len(string)-len(listStr[-1])-len(subStr)

def incremental_backup(container, id, conimg, time_stamp, big, ib_path):
    # define the diff list
    fs_diffadd = []
    fs_diffdelete = []
    fs_diffmodify = []
    container_dir = ib_path + id + "_" + conimg + "_" + time_stamp + "_IB"

    image1 = "daemon://" + conimg + ":" + str(big)
    image2 = "daemon://" + conimg + ":" + time_stamp
    
    # Compare the filesystm difference
    cmd_diffadd = "sudo container-diff diff " + image1 + " " + image2 + " --type=file" + \
        " --format='{{if not .Diff.Adds}}None{{else}}{{range .Diff.Adds}}{{.Name}}{{\"\\n\"}}{{end}}{{end}}'"
    cmd_diffdel = "sudo container-diff diff " + image1 + " " + image2 + " --type=file" + \
        " --format='{{if not .Diff.Dels}}None{{else}}{{range .Diff.Dels}}{{.Name}}{{\"\\n\"}}{{end}}{{end}}'"
    cmd_diffchg = "sudo container-diff diff " + image1 + " " + image2 + " --type=file" + \
        " --format='{{if not .Diff.Mods}}None{{else}}{{range .Diff.Mods}}{{.Name}}{{\"\\n\"}}{{end}}{{end}}'"
    fs_diffadd = fs_difference(cmd_diffadd, fs_diffadd)
    fs_diffdelete = fs_difference(cmd_diffdel, fs_diffdelete)
    fs_diffmodify = fs_difference(cmd_diffchg, fs_diffmodify)

    # create incremental backup folder
    container_IB = ib_path + id + "_" + conimg + "_" + time_stamp + "_IB/"
    os.mkdir(container_IB)
    os.mkdir(container_IB + "Add")
    # os.mkdir(container_IB + "Delete")
    # os.mkdir(container_IB + "Modify")

    # backup the add file
    for add in fs_diffadd:
        # if os.path.isdir(container_dir + fs_diffadd[i]) == True:
        #     pass
        # else:
        add_cmd = "sudo docker cp -a -L " + id + ":" + add + " " + container_IB + "Add"
        print(add_cmd)
        os.popen(add_cmd)
        print(add)
        
    # # back the delete file
    # os.mknod(container_IB + "Delete/delete_list.txt")
    # for i in range(len(fs_diffdelete)):
    #     try:
    #         delete_fp = open(container_IB + "Delete/delete_list.txt", "a")
    #         delete_fp.writelines(fs_diffdelete[i] + "\n")
    #     except IOError as e:
    #         print(e) 
    #     # backup the add file
    # for modify in fs_diffmodify:
    #     save_tar = modify
    #     f = open(save_tar, "wb")
    #     bits, stat = container.get_archive(modify)

    #     for chunk in bits:
    #         f.write(chunk)
    #         f.close()
 
def backup():
    # get all container id
    cmd_conid = os.popen("sudo docker ps -a -q")
    result_conid = cmd_conid.readlines()
    cmd_conid.close()

    backup_path = "/usr/local/Ncsis_Docker_Backup/backup/"

    for id in result_conid:
        # get the current time 
        time_stamp = int(time.time())
        time_stamp = int(time_stamp* (10 ** (10-len(str(time_stamp)))))
        time_stamp = str(time_stamp)

        # reshap the container -d
        id = id[:-1]
        # get the container by container ID.
        container = client.containers.get(id)
        # get the container's "image" attribute
        conimg = container.attrs['Config']['Image']
        
        # create a backup folder for the container
        if os.path.exists(backup_path + id + "_" + conimg ):
            pass
        else:
            os.mkdir(backup_path + id + "_" + conimg)
            os.mkdir(backup_path + id + "_" + conimg + "/full_backup")  
            os.mkdir(backup_path + id + "_" + conimg + "/incremental_backup") 
        
        fb_path = backup_path + id + "_" + conimg + "/full_backup/"
        ib_path = backup_path + id + "_" + conimg + "/incremental_backup/"
        
        # commit the container to image
        container.commit(repository=conimg, tag=time_stamp)

        # get the full backup timestamp
        fb_dirlist = os.listdir(fb_path)
        fb_backuplist = list(filter(lambda x:id + "_" + conimg  in x, fb_dirlist))

        # get the incremental backup list
        ib_dirlist = os.listdir(ib_path)
        ib_backuplist = list(filter(lambda x:id + "_" + conimg  in x, ib_dirlist))

        # if the first time backup 
        if len(fb_backuplist) == 0 and len(ib_backuplist) == 0:
            # do full backup
            full_backup(container, id, conimg, time_stamp, fb_path)
        # if the first time incremental ball
        elif len(fb_backuplist) == 1 and len(ib_backuplist) == 0:
            # do incremental backup
            big = fb_backuplist[0][-14:-4]
            incremental_backup(container, id, conimg, time_stamp, big, ib_path)

            # remove image
            client.images.remove(conimg + ":" + str(big))
        else:
            # get the latest incremental backup timestamp
            for i in range(len(ib_backuplist)):
                big = int(ib_backuplist[i][-13:-3])
                if big < int(ib_backuplist[i][-13:-3]):
                    big == int(ib_backuplist[i][-13:-3])
                else:
                    pass  

            # do incremental backup  
            incremental_backup(container, id, conimg, time_stamp, big, ib_path)

            # remove image
            client.images.remove(conimg + ":" + str(big))

        # avoid the same time_stamp(tag)
        time.sleep(1)

if __name__ == "__main__":
    client = docker.from_env()
    backup()
