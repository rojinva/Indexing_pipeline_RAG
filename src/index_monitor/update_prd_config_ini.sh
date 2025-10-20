#!/bin/bash
 
echo -e "\n -----\n"
currenttime=$(date +'%Y%m%d-%H-%M-%S')
echo "current time is ${currenttime}"
ls
echo -e "\n -----\n"
 
# Setting the subscription context correctly
echo "current subscription context"
environ="prd"
sub_name="sub-lam-${environ}-eng-001"
az account set --subscription "$sub_name"  # Fixing the subscription command
az account show
 
echo -e "\n -----\n"
 
# Fetching secrets from Key Vault
kv_name="kvlamuswoai${environ}eng01" # kvlamuswoaiprdeng01
api_key=$(az keyvault secret show --name "openai-acs-search-service-key" --vault-name "$kv_name" --query "value" -o tsv)
tenant_id=$(az keyvault secret show --name "openai-km-tenant-id" --vault-name "$kv_name" --query "value" -o tsv)
client_id=$(az keyvault secret show --name "openai-km-client-id" --vault-name "$kv_name" --query "value" -o tsv)
client_secret=$(az keyvault secret show --name "openai-km-client-secret" --vault-name "$kv_name" --query "value" -o tsv)
 
# Updating config.ini with the retrieved values
echo -e "\n ------------config.ini file updated with values from $kv_name keyvault-------------- \n"
 
sed -i "/^\[prd\]$/,/^\[/ s/^api_key = .*/api_key = ${api_key}/" src/index_monitor/config.ini
sed -i "/^\[AzureADcredentials-prd\]$/,/^\[/ s/^tenant_id = .*/tenant_id = ${tenant_id}/" src/index_monitor/config.ini
sed -i "/^\[AzureADcredentials-prd\]$/,/^\[/ s/^client_id = .*/client_id = ${client_id}/" src/index_monitor/config.ini
sed -i "/^\[AzureADcredentials-prd\]$/,/^\[/ s|^client_secret = .*|client_secret = ${client_secret}|" src/index_monitor/config.ini
 
#echo -e "\n --------------------\n"
#cat src/index_monitor/config.ini