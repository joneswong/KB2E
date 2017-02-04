#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
using namespace std;

bool debug=false;
bool L1_flag=1;

string version;
string trainortest = "test";

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

int relation_num,entity_num;
int n= 100;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%10==9)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}

class Test{
    vector<vector<double> > relation_vec,entity_vec;
    //TransH
    vector<vector<double> > A;


    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r, fb_lb;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    //for triplet classification
    int cardinality_of_testset;

    void add_labeled_triplet(int x, int y, int z, int lb, bool flag)
    {
        if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
                fb_lb.push_back(lb);
        }
        if (lb) {
            ok[make_pair(x,z)][y]=1;
        }
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    /* double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=-fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=-sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        return sum;
    } */

    //replaced by the version of TransH
    double calc_sum(int e1,int e2,int rel)
    {
        double tmp1=0,tmp2=0;
        for (int jj=0; jj<n; jj++)
        {
        	tmp1+=A[rel][jj]*entity_vec[e1][jj];
            tmp2+=A[rel][jj]*entity_vec[e2][jj];
        }

        double sum=0;
        for (int ii=0; ii<n; ii++)
            sum+=fabs(entity_vec[e2][ii]-tmp2*A[rel][ii]-(entity_vec[e1][ii]-tmp1*A[rel][ii])-relation_vec[rel][ii]);
        return sum;
    }

    void run()
    {
        //TransH in corresponding to the Train_TransH.cpp
        FILE* f1 = fopen(("relation2vec.txt"+version).c_str(),"r");
        FILE* f2 = fopen(("A.txt"+version).c_str(), "r");
        FILE* f3 = fopen(("entity2vec.txt"+version).c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb;i++)
        {
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        A.resize(relation_num_fb);
        for (int i = 0; i < relation_num_fb; i++)
        {
            A[i].resize(n);
            for (int ii = 0; ii < n; ii++)
                fscanf(f1,"%lf",&A[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        fclose(f1);
        fclose(f2);
        fclose(f3);
		double lsum=0 ,lsum_filter= 0;
		double rsum = 0,rsum_filter=0;
		double lp_n=0,lp_n_filter;
		double rp_n=0,rp_n_filter;
		map<int,double> lsum_r,lsum_filter_r;
		map<int,double> rsum_r,rsum_filter_r;
		map<int,double> lp_n_r,lp_n_filter_r;
		map<int,double> rp_n_r,rp_n_filter_r;
		map<int,int> rel_num;

        for (int testid = 0; testid<fb_l.size(); testid+=1)
		{
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
			double tmp = calc_sum(h,l,rel);
			rel_num[rel]+=1;
			vector<pair<int,double> > a;
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(i,l,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			double ttt=0;
			int filter = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(a[i].first,rel)].count(l)>0)
					ttt++;
			    if (ok[make_pair(a[i].first,rel)].count(l)==0)
			    	filter+=1;
				if (a[i].first ==h)
				{
					lsum+=a.size()-i;
					lsum_filter+=filter+1;
					lsum_r[rel]+=a.size()-i;
					lsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=10)
					{
						lp_n+=1;
						lp_n_r[rel]+=1;
					}
					if (filter<10)
					{
						lp_n_filter+=1;
						lp_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			a.clear();
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(h,i,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,rel)].count(a[i].first)>0)
					ttt++;
			    if (ok[make_pair(h,rel)].count(a[i].first)==0)
			    	filter+=1;
				if (a[i].first==l)
				{
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					rsum_r[rel]+=a.size()-i;
					rsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=10)
					{
						rp_n+=1;
						rp_n_r[rel]+=1;
					}
					if (filter<10)
					{
						rp_n_filter+=1;
						rp_n_filter_r[rel]+=1;
					}
					break;
				}
			}
        }
		cout<<"left:"<<lsum/fb_l.size()<<'\t'<<lp_n/fb_l.size()<<"\t"<<lsum_filter/fb_l.size()<<'\t'<<lp_n_filter/fb_l.size()<<endl;
		cout<<"right:"<<rsum/fb_r.size()<<'\t'<<rp_n/fb_r.size()<<'\t'<<rsum_filter/fb_r.size()<<'\t'<<rp_n_filter/fb_r.size()<<endl;
    }

    //for triplet classification
    void run_triplet_classification() {
        //TransH in corresponding to the Train_TransH.cpp
        FILE* f1 = fopen(("relation2vec.txt"+version).c_str(),"r");
        FILE* f2 = fopen(("A.txt"+version).c_str(), "r");
        FILE* f3 = fopen(("entity2vec.txt"+version).c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb;i++)
        {
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        A.resize(relation_num_fb);
        for (int i = 0; i < relation_num_fb; i++)
        {
            A[i].resize(n);
            for (int ii = 0; ii < n; ii++)
                fscanf(f1,"%lf",&A[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        fclose(f1);
        fclose(f2);
        fclose(f3);
        
        //determine thresholds according to valid set
        map<int,vector<pair<int,double> > > a;
        map<int, double> classifier;
        for (int testid = this->cardinality_of_testset; testid<fb_l.size(); testid+=1)
        {
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
                        int lb = fb_lb[testid];
			double tmp = calc_sum(h,l,rel);

                        a[rel].push_back(make_pair(lb, tmp));
        }
        for (map<int, vector<pair<int, double> > >::iterator it = a.begin(); it != a.end(); ++it) {
            int rel = it->first;
            vector<pair<int,double> >& value = it->second;
            sort(value.begin(), value.end(),cmp);
            int num_of_positive_triplets = 0;
            int num_of_triplets = value.size();
            for (vector<pair<int,double> >::iterator pair_it = value.begin(); pair_it != value.end(); ++pair_it) {
                if (pair_it->first) {
                    num_of_positive_triplets++;
                }
            }
            double threshold = 0;
            int num_of_recalled_positive = 0;
            int num_of_correct = num_of_triplets - num_of_positive_triplets;
            for (int i = 0; i < num_of_triplets; ++i) {
                if (value[i].first) {
                    num_of_recalled_positive++;
                }
                int cur_num_of_correct = num_of_recalled_positive + (num_of_triplets-num_of_positive_triplets-(i+1-num_of_recalled_positive));
                if (num_of_correct < cur_num_of_correct) {
                    num_of_correct = cur_num_of_correct;
                    if (i + 1 == num_of_triplets) {
                        threshold = value[i].second + 1e-7;
                    } else {
                        threshold = (value[i].second + value[i+1].second) / 2;
                    }
                }
            }
            classifier[rel] = threshold;
        }

        //classify test triplets by the thresholds
        a.clear();
        for (int testid = 0; testid < this->cardinality_of_testset; testid+=1)
        {
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
                        int lb = fb_lb[testid];
			double tmp = calc_sum(h,l,rel);

                        a[rel].push_back(make_pair(lb, tmp));
        }
        int num_of_considered = 0;
        int num_of_correct = 0;
        for (map<int, vector<pair<int,double> > >::iterator it = a.begin(); it != a.end(); ++it) {
            int rel = it->first;
            map<int, double>::iterator cit = classifier.find(rel);
            if (cit != classifier.end()) {
                double sep = cit->second;
                for(vector<pair<int,double> >::iterator pit = it->second.begin(); pit != it->second.end(); ++pit) {
                    num_of_considered++;
                    if ((pit->first != 0 && pit->second < sep) || (pit->first == 0 && pit->second >= sep)) {
                        num_of_correct++;
                    }
                }
            }
        }

        cout << num_of_correct << "\t" << num_of_considered << "\t" << (double)num_of_correct / (double)num_of_considered << endl;
    }
};
Test test;

void prepare(int has_fourth_column)
{
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		mid2type[st]="None";
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
    string test_file_path = "../data/test.txt";
    //for triplet classification
    if (has_fourth_column) {
        test_file_path = "../data/test_labeled.txt";
    }
    FILE* f_kb = fopen(test_file_path.c_str(), "r");
    test.cardinality_of_testset = 0;
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
        	cout<<"miss relation:"<<s3<<endl;
            relation2id[s3] = relation_num;
            relation_num++;
        }
        
        //for triplet classification
        if (has_fourth_column) {
            fscanf(f_kb, "%s", buf);
            string s4 = buf;
            if (s4.compare("true")==0) {
                test.add_labeled_triplet(entity2id[s1],entity2id[s2],relation2id[s3],1,true);
            } else {
                test.add_labeled_triplet(entity2id[s1],entity2id[s2],relation2id[s3],0,true);
            }
        } else {
            test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
        }
        test.cardinality_of_testset++;
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen("../data/train.txt","r");
	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }

        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    string valid_file_path = "../data/valid.txt";
    //for triplet classification
    if (has_fourth_column) {
        valid_file_path = "../data/valid_labeled.txt";
    }
    FILE* f_kb2 = fopen(valid_file_path.c_str(),"r");
	while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }

        //for triplet classification
        if (has_fourth_column) {
            fscanf(f_kb, "%s", buf);
            string s4 = buf;
            if (s4.compare("true")==0) {
                test.add_labeled_triplet(entity2id[s1],entity2id[s2],relation2id[s3],1,true);
            } else {
                test.add_labeled_triplet(entity2id[s1],entity2id[s2],relation2id[s3],0,true);
            }
        } else {
            test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
        }
    }
    fclose(f_kb2);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        int i;
        if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-version", argc, argv)) > 0) version  = argv[i + 1];
        cout<<"size = "<<n<<endl;
        cout<<"version = "<<version<<endl;

        version = argv[1];
        //link prediction
        //prepare(0);
        //test.run();
        //triplet classification
        prepare(1);
        test.run_triplet_classification();
    }
}

