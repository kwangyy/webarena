#!/usr/bin/env bash
# Reset WebArena sites to their initial state by replacing their containers with
# fresh ones from the original images (environment_docker/README.md, "Environment
# reset"). Everything runs on the docker host over ssh; only the named sites are
# touched, so other containers on that host are left alone.
#
#   jiuwen_arena/reset_sites.sh forum                   # reddit only
#   jiuwen_arena/reset_sites.sh forum shopping shopping_admin
#   WEBARENA_SSH=root@host WEBARENA_HOST=host jiuwen_arena/reset_sites.sh gitlab
#
# Sites: forum (reddit, :9999), shopping (:7770), shopping_admin (:7780),
# gitlab (:8023). Map and Wikipedia are read-only and never need a reset.
# Tested: forum (2026-11-25), shopping and shopping_admin (2026-11-26). The shopping, shopping_admin and gitlab steps follow
# the README but have not been run from this script yet.
set -euo pipefail

SSH_TARGET="${WEBARENA_SSH:-root@157.230.85.206}"
HOST="${WEBARENA_HOST:-${SSH_TARGET#*@}}"   # hostname the sites are served on

if [ $# -eq 0 ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$SSH_TARGET" "$@"; }

# Wait until a URL on the docker host answers 200 (curl failures are retried, not fatal).
# want=any accepts any HTTP answer: a fresh Magento redirects (302) to its old host name
# until its base URL is set, so it never answers 200 before that step.
# host=the name to ask for: once a Magento's base URL is set, it redirects every other host
# name (localhost included) to that one, so the check after setup uses the public name.
wait_ok() {  # name port max_seconds [any|exact] [host]
    remote "for i in \$(seq 1 \$(( $3 / 5 ))); do
                c=\$(curl -s -o /dev/null -w '%{http_code}' http://${5:-localhost}:$2/ || true)
                { [ \"\$c\" = 200 ] || { [ '${4:-}' = any ] && [ \"\$c\" != 000 ]; }; } && { echo '$1 up (http 200) after ~'\$((i*5))'s'; exit 0; }
                sleep 5
            done; echo '$1 NOT up after $3 s (last http '\$c')'; exit 1"
}

replace() {  # container run-args...
    local name=$1; shift
    echo "== $name: replacing container"
    remote "docker stop $name >/dev/null 2>&1 || true; docker rm $name >/dev/null 2>&1 || true; docker run --name $name $*"
}

for site in "$@"; do
    case "$site" in
        forum|reddit)
            replace forum -p 9999:80 -d postmill-populated-exposed-withimg
            wait_ok forum 9999 300
            ;;
        shopping)
            replace shopping -p 7770:80 -d shopping_final_0712
            wait_ok shopping 7770 600 any
            # A fresh Magento does not know this host's name: set the base URLs again (README step 6).
            remote "docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url='http://$HOST:7770' &&
                    docker exec shopping mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value=\"http://$HOST:7770/\" WHERE path = \"web/secure/base_url\";' &&
                    docker exec shopping /var/www/magento2/bin/magento cache:flush"
            wait_ok shopping 7770 300 exact "$HOST"
            ;;
        shopping_admin)
            replace shopping_admin -p 7780:80 -d shopping_admin_final_0719
            wait_ok shopping_admin 7780 600 any
            remote "docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0 &&
                    docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0 &&
                    docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url='http://$HOST:7780' &&
                    docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value=\"http://$HOST:7780/\" WHERE path = \"web/secure/base_url\";' &&
                    docker exec shopping_admin /var/www/magento2/bin/magento cache:flush"
            wait_ok shopping_admin 7780 900 exact "$HOST"
            ;;
        gitlab)
            replace gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
            remote "docker exec gitlab update-permissions &&
                    docker exec gitlab sed -i \"s|^external_url.*|external_url 'http://$HOST:8023'|\" /etc/gitlab/gitlab.rb &&
                    docker exec gitlab gitlab-ctl reconfigure"
            wait_ok gitlab 8023 900
            ;;
        *)
            echo "unknown site: $site (forum, shopping, shopping_admin, gitlab)" >&2
            exit 2
            ;;
    esac
done

# Check from this machine too, the way the runs reach the sites.
for site in "$@"; do
    case "$site" in forum|reddit) p=9999;; shopping) p=7770;; shopping_admin) p=7780;; gitlab) p=8023;; esac
    echo "$site from here: http $(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$p/" || true)"
done
