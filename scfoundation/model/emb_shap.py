import numpy as np
import torch
import shap
from tqdm import tqdm

def trace_important_genes(pretrainmodel, classifier_model, adata, background_samples=100, n_genes=15):
    gene_names = adata.var_names
    gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    with torch.no_grad():
        device = next(pretrainmodel.parameters()).device
        x = torch.tensor(gene_expr).float().to(device)
        position_gene_ids = torch.zeros(x.shape[0], dtype=torch.long).to(device)
        
        x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = pretrainmodel.pos_emb(position_gene_ids)
        x += position_emb
        geneemb = pretrainmodel.encoder(x, torch.zeros_like(x, dtype=torch.bool))
        
        if args.pool_type == 'all':
            geneemb1 = geneemb[:, -1, :]
            geneemb2 = geneemb[:, -2, :]
            geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
            embeddings = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
        else:
            embeddings, _ = torch.max(geneemb, dim=1)
        
        embeddings = embeddings.detach().cpu().numpy()
    
    background = embeddings[np.random.choice(embeddings.shape[0], background_samples, replace=False)]
    explainer = shap.DeepExplainer(classifier_model, torch.tensor(background).float())
    
    test_samples = torch.tensor(embeddings[:1000]).float()
    shap_values = explainer.shap_values(test_samples)
    
    if isinstance(shap_values, list):
        mean_shap = np.mean(np.array(shap_values), axis=(0, 1))
    else:
        mean_shap = np.mean(shap_values, axis=(0,))
    
    sorted_dims = np.argsort(mean_shap)
    bottom_dims = sorted_dims[:n_genes]
    top_dims = sorted_dims[-n_genes:]
    
    token_weights = pretrainmodel.token_emb.weight.detach().cpu().numpy()
    
    def get_gene_scores(dimensions):
        gene_scores = np.zeros(len(gene_names))
        for dim in dimensions:
            weighted_contribution = token_weights[dim] * mean_shap[dim]
            gene_scores += weighted_contribution
        return gene_scores
    
    positive_scores = get_gene_scores(top_dims)
    negative_scores = get_gene_scores(bottom_dims)
    
    top_genes = gene_names[np.argsort(positive_scores)[-n_genes:]]
    bottom_genes = gene_names[np.argsort(negative_scores)[:n_genes]]
    
    if hasattr(pretrainmodel.encoder, 'get_attention_maps'):
        with torch.no_grad():
            attention = pretrainmodel.encoder.get_attention_maps()
            attention_to_genes = attention[:, :, -2:, :-2].mean((0, 1, 2)).cpu().numpy()
        
        positive_scores *= attention_to_genes
        negative_scores *= attention_to_genes
        
        top_genes_attn = gene_names[np.argsort(positive_scores)[-n_genes:]]
        bottom_genes_attn = gene_names[np.argsort(negative_scores)[:n_genes]]
        
        return {
            'top_genes': top_genes,
            'bottom_genes': bottom_genes,
            'top_genes_attention': top_genes_attn,
            'bottom_genes_attention': bottom_genes_attn,
            'shap_values': mean_shap,
            'important_dims': {'top': top_dims, 'bottom': bottom_dims}
        }
    
    return {
        'top_genes': top_genes,
        'bottom_genes': bottom_genes,
        'shap_values': mean_shap,
        'important_dims': {'top': top_dims, 'bottom': bottom_dims}
    }

if __name__ == "__main__":
    results = trace_important_genes(
        pretrainmodel=pretrainmodel,
        classifier_model=classifier_model,
        adata=adata,
        n_genes=15
    )
    
    print("Top 15 positively influential genes:")
    print(results['top_genes'])
    print("\nBottom 15 negatively influential genes:")
    print(results['bottom_genes'])
    
    if 'top_genes_attention' in results:
        print("\nTop genes considering attention:")
        print(results['top_genes_attention'])
        print("\nBottom genes considering attention:")
        print(results['bottom_genes_attention'])