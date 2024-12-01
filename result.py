import matplotlib.pyplot as plt
import numpy as np

log_file_path = 'training/dev=0.02/log.dat'

epochs = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []

processed_epochs = set()

with open(log_file_path, 'r') as file:
    for line in file:
        values = line.strip().split()
        epoch = int(values[0])

        # ignore duplicated epochs, 단위 epoch 마다 그래프 그리도록
        if epoch % 3 != 0 or epoch in processed_epochs:
            continue
        
        tr_loss = float(values[1])
        tr_acc = float(values[2])
        tst_loss = float(values[9])
        tst_acc = float(values[10])
        
        epochs.append(epoch)
        train_loss.append(tr_loss)
        train_acc.append(tr_acc * 100)  
        test_loss.append(tst_loss)
        test_acc.append(tst_acc * 100) 
        
        processed_epochs.add(epoch)

sorted_indices = np.argsort(epochs)
epochs = np.array(epochs)[sorted_indices]
train_loss = np.array(train_loss)[sorted_indices]
train_acc = np.array(train_acc)[sorted_indices]
test_loss = np.array(test_loss)[sorted_indices]
test_acc = np.array(test_acc)[sorted_indices]

fig, ax1 = plt.subplots()

# Loss 플롯
# ax1.plot(epochs, train_loss, label="Train Loss", color="blue", alpha=0.7)
ax1.plot(epochs, test_loss, label="Test Loss", color="red", alpha=0.8, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper left')

# Accuracy 플롯
ax2 = ax1.twinx()
# ax2.plot(epochs, train_acc, label="Train Accuracy", color="green", linestyle="--", alpha=0.7)
ax2.plot(epochs, test_acc, label="Test Accuracy", color="blue", alpha=0.8, linewidth=2)
ax2.set_ylabel('Accuracy (%)')
ax2.legend(loc='lower right')


# 그래프 제목과 출력
plt.title('Training and Testing Loss/Accuracy')
plt.tight_layout()
plt.show()
