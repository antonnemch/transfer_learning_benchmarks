from matplotlib import pyplot as plt


def vis_lrs(layer_lrs, fold):
    layer_names = ['Conv 1', 'Block 1-1', 'Block 1-2', 'Block 2-1', 'Block 2-2',
                   'Block 3-1', 'Block 3-2', 'Block 4-1', 'Block 4-2', 'FC']
    layer_lrs = list(zip(*layer_lrs))
    
    plt.figure(figsize=(7,4), dpi=200)
    for i in range(len(layer_names)):
        plt.plot(layer_lrs[i], label=layer_names[i])
    num1 = 1.02
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.title('LRs on POCUS Pneumonia Detection Task (fold {})'.format(fold))
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.tight_layout()
    
    plt.savefig('lr_curve_pocus_fold{}.svg'.format(fold), format='svg', dpi=200)
    plt.show()