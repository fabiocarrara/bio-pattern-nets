import os
import math
from base64 import b64encode

import jinja2
import torch

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from io import BytesIO
from scipy.misc import imsave
from tqdm import tqdm
from torch.autograd import Variable as V


def score(model, loader, out, args):
    model.train(False)

    dataset_size = len(loader.dataset)
    tot_iterations = math.ceil(dataset_size / float(args.batchSize))
    tot_iterations = int(tot_iterations)

    # Iterate over data.
    # score_fname = os.path.join(args.workDir, out)
    scores = torch.Tensor(dataset_size, 1)
    i = 0
    # with open(score_fname, 'wb') as f:
    for iteration, data in tqdm(enumerate(loader), desc='Iterations',
                                total=tot_iterations):
        inputs, labels = data
        inputs = V(inputs.cuda(), volatile=True)
        outputs = model(inputs)
        n_samples = outputs.size()[0]
        scores[i:(i+n_samples)] = outputs.data
        i += n_samples
        #for o, l in zip(outputs, labels):
        #    f.write('{}\t{}\n'.format(o.data[0], l))

    # print 'Scores saved:', score_fname
    return scores


def evaluate(model, data, args):
    # SCORING -----------
    train_loader, val_loader, test_loader = data

    train_scores = score(model, train_loader, 'train_scores.tsv', args)
    val_scores = score(model, val_loader, 'val_scores.tsv', args)
    test_scores = score(model, test_loader, 'test_scores.tsv', args)
    
    test_scores = test_scores.squeeze()

    sorted_scores, idx = torch.sort(test_scores, 0, descending=True)
    sorted_images = test_loader.dataset.data_tensor[idx]
    sorted_labels = test_loader.dataset.target_tensor[idx]
    
    n_labels = len(set(sorted_labels))
    sorted_labels = n_labels - 1 - sorted_labels
    sorted_labels = [chr(ord('A') + i) for i in sorted_labels]

    def to_base64(image):
        buf = BytesIO()
        imsave(buf, image.numpy(), format='png')
        return b64encode(buf.getvalue()).decode('utf8')

    # PLOT SCORES
    sns.set()
    plt.figure(figsize=(16,8))
    plot_data = pd.DataFrame({'score': sorted_scores.numpy(), 'label': sorted_labels})
    sns.violinplot(data=plot_data, x='label', y='score', inner='points')
    plt.title('Distribution of scores over labels')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plot_data = b64encode(buf.getvalue()).decode('utf8')

    # BUILD WEB PAGE
    sorted_images = sorted_images.permute(0, 2, 3, 1) # to NHWC
    sorted_images = [to_base64(i) for i in sorted_images]

    results = zip(sorted_images, sorted_scores, sorted_labels)

    html_template = jinja2.Template('''
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Ordinal Regression Evaluation</title>
      <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" rel="stylesheet"/>
      <link href="https://cdnjs.cloudflare.com/ajax/libs/datatables/1.10.12/css/dataTables.bootstrap.min.css" rel="stylesheet"/>
    </head>
    <body>
    <div class="container">
        <h1>EDGES dataset: ordinal regression results</h1>
        <figure class>
            <img style="width:100%" src="data:image/png;base64,{{plot}}">
        </figure>
        <table id="data" class="table table-striped table-bordered table-hover" cellspacing="0" width="100%">
            <thead><tr><th>Image</th><th>Score</th><th>Label</th></tr></thead>
            <tbody>
            {% for i, s, l in results %}
                <tr><td><img src="data:image/png;base64,{{i}}"></td><td>{{s}}</td><td>{{l}}</td></tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/datatables/1.10.12/js/jquery.dataTables.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/datatables/1.10.12/js/dataTables.bootstrap.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#data').DataTable({
                "lengthMenu": [[10, 50, 100, 500, -1], [10, 50, 100, 500, "All"]]
            });
        });
    </script>
    </body>
    </html>
    ''')

    results_fname = 'test_results.html'
    results_fname = os.path.join(args.workDir, results_fname)
    with open(results_fname, 'wb') as f:
        f.write(html_template.render(results=results, plot=plot_data))

    print "Results written:", results_fname
