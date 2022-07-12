SUBDIRS = format op_ op_uni

ifeq ($(HIP_ON),1)
SUBDIRS += op_dcu hint
endif
ifeq ($(ARM_ON),1)
SUBDIRS += op hint
endif
ifeq ($(HYGON_ON),1)
SUBDIRS += op
endif
.PHONY : lib

lib : 
	@for d in $(SUBDIRS) ; \
	do if test -d $$d; then \
	  $(MAKE) -C $$d $(@F) || exit 1 ; \
	fi; \
	done
