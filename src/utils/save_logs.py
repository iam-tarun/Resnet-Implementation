def save_metrics_to_log(epoch, loss, accuracy, precision, recall, f1, log_file):
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n')