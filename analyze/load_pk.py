import pickle as pk
import os, sys
import numpy as np
import matplotlib.pyplot as plt
# with open('/storage/UDA/UDA2/exp_office/amazon_dslr/0.001_1_0.1/Dec08_18-40-41/output.pkl', 'rb') as f:
#     epoch_results, best_results = pk.load(f)

domain = {'exp_office': ['amazon', 'dslr', 'webcam'],
          'exp_officehome': ['Art', 'Clipart', 'Product', 'Real_World']}
root = '/storage/UDA/UDA2/lr_oh_0.01_o_0.001_2'
hyp = '0.001_1_0.0'
for key, val in domain.items():
    for s in val:
        for t in val:
            if s != t:
                subfd = os.path.join(root, key, s+'_'+t, hyp)
                entries = os.listdir(subfd)
                subfolders = [os.path.join(subfd, entry) for entry in entries if os.path.isdir(os.path.join(subfd, entry))]
                fd = subfolders[0]
                fd = os.path.join(fd, 'output.pkl')
                with open(fd, 'rb') as f:
                    epoch_results, best_results = pk.load(f)
                    #print(fd, epoch_results)
                    epochs, lcloss, lmloss, lloss, lhos, lacc, lnmi = [],[],[],[],[],[], []
                    for epoch, metrics in epoch_results.items():
                        print(metrics)
                        closs_total=metrics['closs_total']
                        mloss_total=metrics['mloss_total']
                        loss_total=metrics['loss_total']
                        if 'hos' not in metrics.keys():
                            hos = np.nan
                        else:
                            hos = metrics['hos']
                        if 'acc_test' not in metrics.keys():
                            acc_test = np.nan
                        else:
                            acc_test = metrics['acc_test']

                        if 'nmi' not in metrics.keys():
                            nmi = np.nan
                        else:
                            nmi=metrics['nmi']

                        epochs.append(epoch)
                        lcloss.append(closs_total)
                        lmloss.append(mloss_total)
                        lloss.append(loss_total)
                        lhos.append(hos)
                        lnmi.append(nmi)
                        lacc.append(acc_test)

                    plt.figure(figsize=(8, 6))
                    lcloss = np.array(lcloss)
                    lcloss = lcloss / np.max(lcloss)
                    lmloss = np.array(lmloss)
                    lmloss = lmloss / np.max(lmloss)
                    lloss = np.array(lloss)
                    lloss = lloss / np.max(lloss)
                    print(lcloss)
                    # Plot each y-axis vector
                    plt.plot(epochs, lcloss, label='closs', color='blue')
                    plt.plot(epochs, lmloss, label='mloss', color='green')
                    plt.plot(epochs, lloss, label='loss', color='purple')
                    plt.plot(epochs, lhos, label='hos', color='red')
                    plt.plot(epochs, lnmi, label='nmi', color='black')
                    plt.plot(epochs, lacc, label='acc', color='orange')
                    # Add labels, legend, and title
                    plt.xlabel('X-axis')
                    plt.ylabel('Y-axis')
                    #plt.title('Line Plot with Multiple Y Vectors')
                    plt.legend()  # Display legend
                    plt.grid(True)  # Add a grid for better readability

                    # Show the plot
                    plt.savefig('{}_{}_{}.png'.format(key,s,t), dpi=300, bbox_inches='tight')  # Save as PNG with high resolution